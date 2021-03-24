from typing import Dict, List, Tuple
import cv2
import numpy as np
import torch
torch.set_grad_enabled(False) # Instead of with torch.no_grad
import time
import torchvision
import torch.nn.functional as F

from net import PriorBox, decode, decode_batch, decode_landm, decode_landm_batch, apply_indices_2d_to_3d, py_cpu_nms
from retinaface import RetinaFace


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


class RetinaNetDetector(torch.jit.ScriptModule):
# class RetinaNetDetector():
    def __init__(self):
        super(RetinaNetDetector, self).__init__()
        # Config world
        self.cfg_variance: List[float] = [0.1, 0.2]
        self.cfg_min_sizes: List[List[int]] = [[16, 32], [64, 128], [256, 512]]
        self.cfg_steps: List[int] = [8, 16, 32]
        self.cfg_clip: bool = False

        if torch.cuda.is_available(): 
            self.device = 'cuda'
        else: 
            self.device = 'cpu'
        self.model = RetinaFace()
        self.model.eval()
        self.device = self.device
        print("Device:", self.device)
        self.resize = 1
        self.confidence_threshold = 0.9
        self.top_k = 5000
        self.nms_threshold = 0.4
        self.keep_top_k = 750
        self.threshold = 0.9
        self.mean_tensor = torch.tensor([104, 117, 123], device=self.device)
        self.prior_data: Dict[str, torch.Tensor] = {'0x0': torch.Tensor([1])}

    @torch.jit.ignore
    def load_weights(self, weight_path):
        print('Loading pretrained model from {}'.format(weight_path))
        pretrained_dict = torch.load(
            weight_path, map_location=torch.device('cpu'))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = remove_prefix(
                pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = remove_prefix(pretrained_dict, 'module.')
        check_keys(self.model, pretrained_dict)
        self.model.load_state_dict(pretrained_dict, strict=False)
        self.model.eval()
        self.model.to(self.device)

    @torch.jit.script_method
    def preprocess(self, 
                   image: torch.Tensor,
                   target_size: int = -1):
        img = image.to(self.device).float()

        # Mimic torchvision.transforms.ToTensor to use interpolate
        img = img.permute(2, 0, 1).unsqueeze(0)
        actual_scale: float = 1.
        if target_size != -1:
            image_size = img.size()
            image_height, image_width = int(image_size[-2]), int(image_size[-1])
            if max(image_height, image_width) > target_size:
                if image_height >= image_width:
                    actual_scale = target_size / image_height
                    new_height = target_size
                    new_width = int(image_width * actual_scale)
                else:
                    actual_scale = target_size / image_width
                    new_width = target_size
                    new_height = int(image_height * actual_scale)
                img = F.interpolate(img, size=(new_height, new_width), mode="bicubic", align_corners=False)
        
        img = img.squeeze(0).permute(1, 2, 0)
        img -= self.mean_tensor  # mean substraction
        img = img.permute(2, 0, 1).unsqueeze(0)
        img_shape = img.size()
        im_height, im_width = int(img_shape[2]), int(img_shape[3])
        scale = torch.tensor(
            [im_width, im_height, im_width, im_height], device=self.device)
        return img, scale, im_height, im_width, actual_scale

    @torch.jit.script_method
    def postprocess(self, 
                    img: torch.Tensor, 
                    im_height: int, 
                    im_width: int, 
                    scale: torch.Tensor, 
                    loc: torch.Tensor,
                    conf: torch.Tensor, 
                    landms: torch.Tensor,
                    actual_scale: float):
        image_size_str = f"{im_height}x{im_width}"
        if image_size_str not in self.prior_data.keys():
            priorbox = PriorBox(min_sizes=self.cfg_min_sizes,
                                steps=self.cfg_steps,
                                clip=self.cfg_clip,
                                image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(self.device)
            prior_data = priors.data
            self.prior_data[image_size_str] = prior_data
        else:
            prior_data = self.prior_data[image_size_str]
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg_variance)
        boxes = boxes * scale / self.resize

        # NOTE: Trying to get out of numpy from here
        scores = conf.squeeze(0)[:, 1]
        landms = decode_landm(landms.data.squeeze(
            0), prior_data, self.cfg_variance)
        scale1 = torch.tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]], device=self.device)
        landms = landms * scale1 / self.resize

        # ignore low scores
        # NOTE: TorchScript raised TypeError: module, class, method, function, traceback, frame, or code object was expected, got builtin_function_or_method
        # Debugging show that it's breaking due to np.where
        inds = torch.where(scores > self.confidence_threshold)
        inds = inds[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]


        # keep top-K before NMS
        order = torch.argsort(scores, descending=True)[:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = torch.hstack((boxes, torch.unsqueeze(scores, dim=1)))
        keep = py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        landms = landms[:self.keep_top_k, :]
        dets = torch.cat((dets, landms), dim=1)

        temp = []
        for x in dets:
            if x[4] >= self.threshold:
                temp.append(x)

        output: Tuple[List[torch.Tensor], List[torch.Tensor]] = ([], [])
        for x in temp:
            x[0:4] = x[0:4] / actual_scale
            output[0].append(x[0:5])
            output[1].append(torch.cat((x[5::2], x[6::2]))/actual_scale)
        return output

    @torch.jit.script_method
    def forward(self,
                image: torch.Tensor,
                target_size: int = -1):
        # Preprocessing
        # st = time.time()
        img, scale, im_height, im_width, actual_scale = self.preprocess(
            image, target_size)
        # torch.cuda.synchronize()
        # print("Preprocessing tooks:", (time.time()-st)*1000, "ms")

        # Forward pass
        # st = time.time()
        loc, conf, landms = self.model(img)
        # torch.cuda.synchronize()
        # print("Forward tooks:", (time.time()-st)*1000, "ms")

        # Post processing
        # st = time.time()
        output = self.postprocess(
            img, im_height, im_width, scale, loc, conf, landms, actual_scale)
        # torch.cuda.synchronize()
        # print("Post processing tooks:", (time.time()-st)*1000, "ms")
        return output

    @torch.jit.script_method
    def preprocess_batch(self,
                         images: List[torch.Tensor],
                         target_size: int = 640):
        # Resize on the longer side of the image to the target size (keep aspect ratio)
        batch_size = len(images)
        
        resized_images: List[torch.Tensor] = []
        all_height: List[int] = []
        all_width: List[int] = []
        all_scale: List[float] = []
        for image in images:
            image = image.permute(2, 0, 1).unsqueeze(0)
            image = image.to(self.device).float() # NOTE: LOL, forget to do this lol
            image_size = image.size()
            scale: float = 1.
            image_height, image_width = int(image_size[-2]), int(image_size[-1])
            if image_height >= image_width:
                scale = target_size / image_height
                new_height = target_size
                new_width = int(image_width * scale)
            else:
                scale = target_size / image_width
                new_width = target_size
                new_height = int(image_height * scale)

            # Mimic torchvision.transforms.ToTensor to use interpolate
            resized_image = F.interpolate(image, size=(
                new_height, new_width), mode="bicubic", align_corners=False)
            resized_image = resized_image.squeeze(0).permute(1, 2, 0)
            resized_image -= self.mean_tensor  # mean substraction
            resized_image = resized_image.permute(2, 0, 1)
            resized_images.append(resized_image) # Drop the frist dims lol
            all_width.append(new_width)
            all_height.append(new_height)
            all_scale.append(scale)

        # Zero padding
        # st = time.time()
        max_width = max(all_width)
        max_height = max(all_height)
        batched_tensor = torch.zeros((batch_size, 3, max_height, max_width), device=self.device).float()
        for index in range(batch_size):
            resized_image = resized_images[index]
            image_size = resized_image.size()
            # print('At index', index, 'size is', image_size)
            image_height, image_width = int(image_size[1]), int(image_size[2])
            batched_tensor[index, :, :image_height,
                           :image_width] = resized_image
        return batched_tensor, all_scale

    @torch.jit.script_method
    def postprocess_batch(self,
                        batched_tensor: torch.Tensor,
                        all_scale: List[float],
                        loc: torch.Tensor,
                        conf: torch.Tensor,
                        landms: torch.Tensor):
        # st = time.time()
        batch_size, channels, im_height, im_width = batched_tensor.shape
        image_size_str = f"{im_height}x{im_width}"
        if image_size_str not in self.prior_data.keys():
            # print("Not go in here")
            priorbox = PriorBox(min_sizes=self.cfg_min_sizes,
                                steps=self.cfg_steps,
                                clip=self.cfg_clip,
                                image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(self.device)
            prior_data = priors.data
            self.prior_data[image_size_str] = prior_data
        else:
            # print("Prior data already exist")
            # torch.cuda.synchronize()
            # st4 = time.time()
            prior_data = self.prior_data[image_size_str]
            # torch.cuda.synchronize()
            # print("Get prior_data actually tooks:", (time.time()-st4)*1000, "ms")
        # torch.cuda.synchronize()
        # print("Get prior_data tooks:", (time.time()-st)*1000, "ms")


        # st2 = time.time()
        boxes = decode_batch(loc.data, prior_data, self.cfg_variance)
        # torch.cuda.synchronize()
        # print("Decode boxes tooks:", (time.time()-st2)*1000, "ms")
        scale = torch.tensor(
            [im_width, im_height, im_width, im_height], device=self.device)
        boxes = boxes * scale / self.resize

        scores = conf[:, :, 1]
        # st3 = time.time()
        landms = decode_landm_batch(landms.data, prior_data, self.cfg_variance)
        # torch.cuda.synchronize()
        # print("Decode lanmarks tooks:", (time.time()-st3)*1000, "ms")
        scale1 = torch.tensor([im_width, im_height, im_width, im_height,
                               im_width, im_height, im_width, im_height,
                               im_width, im_height], device=self.device)
        landms = landms * scale1 / self.resize
        # torch.cuda.synchronize()
        # print("Decode result tooks:", (time.time()-st)*1000, "ms")


        # ignore low scores
        # NOTE: Since the threshold operation might result in different
        # number of remained candidates for each sample in the batch,
        # we actually cannot do it in batch but still need to threshold
        # for each sample, then find a ways to batch them up and use
        # batched nms from torchvision
        masks = scores > self.confidence_threshold
        thresholded_boxes = []
        thresholded_landmarks = []
        thresholded_scores = []
        max_candidates_size = 0
        for index in range(batch_size):
            thresholded_boxes.append(boxes[index][masks[index]])
            thresholded_landmarks.append(landms[index][masks[index]])
            new_score = scores[index][masks[index]]
            thresholded_scores.append(new_score)
            candidate_size = new_score.size()[0]
            if candidate_size > max_candidates_size:
                max_candidates_size = candidate_size

        # NOTE: Suddenly, I got different result for each sample of the batch after
        # thresholding. But since I'm sending the same image in the batch,I expect
        # the same result for each sample in the batch. Must investigate why ?.
        # Turn out the preprocess_batch function had a bug in it. I had fixed the bug
        # And it appear to be working again

        # Now what we need to do is to pad all candidate into a single batch for
        # sorting, trimming, batched NMS, and trimming again.
        # The important thing here is to keep track of what is the padded
        # sample inside the batch and discard it later on.
        padded_thresholded_boxes = []
        padded_thresholded_landmarks = []
        padded_thresholded_scores = []
        for index in range(batch_size):
            candidate_size = thresholded_scores[index].size()[0]
            # if candidate_size < max_candidates_size:
            # Pad box tensor
            box = thresholded_boxes[index]
            padded_box = torch.zeros(
                (max_candidates_size, 4), dtype=box.dtype, device=box.device)
            padded_box[:candidate_size, :] = box
            padded_thresholded_boxes.append(padded_box)

            # Pad landmark tensor
            landmark = thresholded_landmarks[index]
            padded_landmark = torch.zeros(
                (max_candidates_size, 10), dtype=landmark.dtype, device=landmark.device)
            padded_landmark[:candidate_size, :] = landmark
            padded_thresholded_landmarks.append(padded_landmark)

            # Pad score tensor
            score = thresholded_scores[index]
            padded_score = torch.zeros(
                (max_candidates_size), dtype=score.dtype, device=score.device)
            padded_score[:candidate_size] = score
            padded_thresholded_scores.append(padded_score)

        # Now stack them up cause they had the same size babyyy
        new_boxes = torch.stack(padded_thresholded_boxes)
        new_landms = torch.stack(padded_thresholded_landmarks)
        new_scores = torch.stack(padded_thresholded_scores)

        # keep top-K before NMS
        order = torch.argsort(new_scores, descending=True)[:, :self.top_k]
        new_boxes = apply_indices_2d_to_3d(order, new_boxes)
        new_landms = apply_indices_2d_to_3d(order, new_landms)
        new_scores = new_scores.gather(1, order)

        # Now do batched NMS
        # Flatten before nms
        indices = torch.arange(batch_size, device=boxes.device)
        indices = indices[:, None].expand(
            batch_size, max_candidates_size).flatten()
        boxes_flat = new_boxes.flatten(0, 1)
        landmark_flat = new_landms.flatten(0, 1)
        scores_flat = new_scores.flatten()
        batch_index_flat = torch.arange(
            batch_size, device=boxes.device).unsqueeze(-1).expand(-1, max_candidates_size).flatten()

        indices_flat = torchvision.ops.boxes.batched_nms(
            boxes_flat, scores_flat, indices, iou_threshold=self.nms_threshold)

        kept_boxes = boxes_flat[indices_flat]
        kept_scores = scores_flat[indices_flat]
        kept_landmark = landmark_flat[indices_flat]
        kept_batch_index = batch_index_flat[indices_flat]

        # indices within range(0, N) belongs to img1, and indices within range(N, 2N) belongs to img2
        outputs: List[Tuple[List[torch.Tensor], List[torch.Tensor]]] = []
        for id_batch in range(batch_size):
            bbox_of_this_sample: List[torch.Tensor] = []
            score_of_this_samples: List[torch.Tensor] = []
            landmark_of_this_samples: List[torch.Tensor] = []

            # Gather output for each sample in the batch here
            for idx, ind_box in enumerate(kept_batch_index):
                if int(ind_box) == id_batch and kept_scores[idx] >= self.threshold:
                    bbox_of_this_sample.append(kept_boxes[idx])
                    score_of_this_samples.append(kept_scores[idx])
                    landmark_of_this_samples.append(kept_landmark[idx])

            current_sample_output: Tuple[List[torch.Tensor], List[torch.Tensor]] = ([], [])
            if len(bbox_of_this_sample) > 0:
                bbox = torch.stack(bbox_of_this_sample)
                score = torch.stack(score_of_this_samples)
                landmark = torch.stack(landmark_of_this_samples)

                # keep top-K faster NMS
                bbox = torch.hstack(bbox, torch.unsqueeze(score, dim=1))
                bbox = bbox[:self.keep_top_k, :]
                landmark = landmark[:self.keep_top_k, :]
                res = torch.cat((bbox, landmark), dim=1)

                # Scale back
                actual_scale: float = all_scale[id_batch]
                for each in res:
                    each[0:4] = each[0:4] / actual_scale
                    current_sample_output[0].append(each[0:5])
                    current_sample_output[1].append(
                        torch.cat((each[5::2], each[6::2]))/actual_scale)
            outputs.append(current_sample_output)

        return outputs

    @torch.jit.script_method
    def forward_batch(self, 
                      images: List[torch.Tensor],
                      target_size: int = 640):
        # NOTE: Target size for max length side

        # Preprocessing
        # st = time.time()
        batched_tensor, all_scale = self.preprocess_batch(images, target_size)
        # torch.cuda.synchronize()
        # print("Preprocessing tooks:", (time.time()-st)*1000, "ms")

        # Forward pass
        # st = time.time()
        loc, conf, landms = self.model(batched_tensor)
        # torch.cuda.synchronize()
        # print("Forward tooks:", (time.time()-st)*1000, "ms")


        # Postprocessing
        # st = time.time()
        output = self.postprocess_batch(batched_tensor, all_scale, loc, conf, landms)
        # torch.cuda.synchronize()
        # print("Postprocess tooks:", (time.time()-st)*1000, "ms")
        return output

# NOTE: New finding. When turn on script mode for only preprocess batch and post process batch
# (excluding forward_batch), we achive 30ms less inference time ???? => You forgot the warmup
# all the batching function, you dumb ass
