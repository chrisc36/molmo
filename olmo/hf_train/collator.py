from typing import Dict, Any, List

import torch
import torch.nn.functional as F


class HF_MMCollator:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
    
        
    # Helper to pad and stack batches safely
    def pad_and_stack(self, batch_list, pad_value):
        padding_side = getattr(self.tokenizer, "padding_side", "right")
        # First pad sub-batches individually
        sub_batches = []
        for sub_batch in batch_list:
            if padding_side == "right":
                padded = torch.nn.utils.rnn.pad_sequence(sub_batch, batch_first=True, padding_value=pad_value)
            else:  # left padding
                max_len = max(t.size(0) for t in sub_batch)
                padded = torch.stack([
                    F.pad(t, (max_len - t.size(0), 0), value=pad_value) for t in sub_batch
                ])
            sub_batches.append(padded)

        # Find global max length
        max_len = max(t.shape[1] for t in sub_batches)

        # Pad all sub-batches to global max
        padded = []
        for t in sub_batches:
            if t.shape[1] < max_len:
                if padding_side == "right":
                    t = F.pad(t, (0, max_len - t.shape[1]), value=pad_value)
                else:
                    t = F.pad(t, (max_len - t.shape[1], 0), value=pad_value)
            padded.append(t)

        return torch.cat(padded, dim=0)
        
    
    def create_custom_attention_mask(self, attention_mask, text_part_indices, seq_length):
        # Get dimensions
        batch_size = attention_mask.shape[0]
        device = attention_mask.device
        dtype = torch.float32
        min_dtype = float('-inf')
        
        # Create causal mask (similar to first implementation)
        causal_mask = torch.full((seq_length, seq_length), fill_value=min_dtype, dtype=dtype, device=device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        
        # Expand to batch dimension and broadcast
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        
        # Incorporate padding mask from attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = attention_mask[:, None, None, :]
            
            # Combine causal mask with padding mask
            # Where attention_mask is 0, we set to min_dtype (no attention)
            causal_mask_padded = causal_mask[:, :, :, :mask_length]
            padding_mask = (causal_mask_padded + padding_mask) == 0
            causal_mask[:, :, :, :mask_length] = causal_mask_padded.masked_fill(padding_mask, min_dtype)
        
        # Apply text_part_indices modifications
        if text_part_indices is not None:
            causal_mask = causal_mask.clone()
            
            flat_indices = []
            indices_mapping = {}
            current_idx = 0
            
            # Create mapping from each instance to its position in the flattened batch
            for batch_idx, sub_batch in enumerate(text_part_indices):
                for i, indices in enumerate(sub_batch):
                    indices_mapping[(batch_idx, i)] = current_idx
                    flat_indices.append(indices)
                    current_idx += 1
            
            # Apply the custom masking for each instance
            for batch_idx in range(len(flat_indices)):
                for entry in flat_indices[batch_idx]:
                    assert len(entry) == 3, "Each entry should have exactly 3 indices"
                    image_start, image_end, second_part_start = entry
                    
                    # Ensure the indices are within the sequence bounds
                    image_end = min(image_end, seq_length)
                    second_part_start = min(second_part_start, seq_length)
                    
                    # Tokens after second_part_start should not attend to tokens between image_end and second_part_start
                    if second_part_start < seq_length and image_end < second_part_start:
                        # Set attention to min_dtype for the specified region
                        causal_mask[batch_idx, :, second_part_start:, image_end:second_part_start] = min_dtype
        
        return causal_mask

    def pad_and_stack_image_entities(self, entity_list, pad_value=0):
        if not entity_list:
            return None
            
        # Flatten the nested list for easier processing
        flat_entities = [entity for sub_batch in entity_list for entity in sub_batch]
        
        if not isinstance(flat_entities[0], torch.Tensor):
            raise NotImplementedError("Non-tensor entities are not supported")
        
        # Get specs from first entity
        device = flat_entities[0].device
        dtype = flat_entities[0].dtype
        
        # Handle different tensor shapes appropriately
        if len(flat_entities[0].shape) == 1:
            # 1D tensor case
            max_dim = max(entity.shape[0] for entity in flat_entities)
            padded_entities = torch.full((len(flat_entities), max_dim), fill_value=pad_value, 
                                      dtype=dtype, device=device)
            
            for i, entity in enumerate(flat_entities):
                dim = entity.shape[0]
                padded_entities[i, :dim] = entity
        
        elif len(flat_entities[0].shape) == 2:
            # 2D tensor case (like pooled_patches_idx)
            max_dim1 = max(entity.shape[0] for entity in flat_entities)
            max_dim2 = max(entity.shape[1] for entity in flat_entities)
            
            padded_entities = torch.full((len(flat_entities), max_dim1, max_dim2), 
                                      fill_value=pad_value, dtype=dtype, device=device)
            
            for i, entity in enumerate(flat_entities):
                dim1, dim2 = entity.shape
                padded_entities[i, :dim1, :dim2] = entity
        
        elif len(flat_entities[0].shape) == 3:
            # 3D tensor case (common for image features)
            max_dim1 = max(entity.shape[0] for entity in flat_entities)
            max_dim2 = max(entity.shape[1] for entity in flat_entities)
            max_dim3 = max(entity.shape[2] for entity in flat_entities)
            
            padded_entities = torch.full((len(flat_entities), max_dim1, max_dim2, max_dim3), 
                                      fill_value=pad_value, dtype=dtype, device=device)
            
            for i, entity in enumerate(flat_entities):
                dim1, dim2, dim3 = entity.shape
                padded_entities[i, :dim1, :dim2, :dim3] = entity
        
        elif len(flat_entities[0].shape) == 4:
            # 4D tensor case (for RGB images with channels, etc.)
            max_dim1 = max(entity.shape[0] for entity in flat_entities)
            max_dim2 = max(entity.shape[1] for entity in flat_entities)
            max_dim3 = max(entity.shape[2] for entity in flat_entities)
            max_dim4 = max(entity.shape[3] for entity in flat_entities)
            
            padded_entities = torch.full((len(flat_entities), max_dim1, max_dim2, max_dim3, max_dim4), 
                                      fill_value=pad_value, dtype=dtype, device=device)
            
            for i, entity in enumerate(flat_entities):
                dim1, dim2, dim3, dim4 = entity.shape
                padded_entities[i, :dim1, :dim2, :dim3, :dim4] = entity
        
        else:
            # Handle any other dimension case generically
            # Get max dims for each dimension
            num_dims = len(flat_entities[0].shape)
            max_dims = []
            for dim_idx in range(num_dims):
                max_dims.append(max(entity.shape[dim_idx] for entity in flat_entities))
            
            # Create output tensor shape with batch dim
            out_shape = (len(flat_entities),) + tuple(max_dims)
            padded_entities = torch.full(out_shape, fill_value=pad_value, 
                                      dtype=dtype, device=device)
            
            # Fill with actual data
            for i, entity in enumerate(flat_entities):
                # Create dynamic slicing based on entity dimensions
                slices = [slice(0, entity.shape[dim_idx]) for dim_idx in range(num_dims)]
                padded_entities[(i,) + tuple(slices)] = entity
                
        return padded_entities
    
    def __call__(self, batch: List[List[Dict[str, Any]]]) -> Dict[str, Any]:

        if "text_part_indices" in batch[0][0]:
            text_part_indices = [[instance["text_part_indices"] for instance in sub_batch] for sub_batch in batch]
        else:
            text_part_indices = None
        

        if "labels" in batch[0][0]:
            # Extract and pad
            input_ids_nested, labels_nested, loss_masks_nested = tuple(
                [[instance[key] for instance in sub_batch] for sub_batch in batch]
                for key in ("input_ids", "labels", "loss_masks")
            )
        else:
            # Extract and pad
            input_ids_nested, loss_masks_nested = tuple(
                [[instance[key] for instance in sub_batch] for sub_batch in batch]
                for key in ("input_ids", "loss_masks")
            )
            labels_nested = None

        input_ids = self.pad_and_stack(input_ids_nested, pad_value=self.tokenizer.pad_token_id)
        if labels_nested is not None:
            labels = self.pad_and_stack(labels_nested, pad_value=-100)
        else:
            labels = None
        loss_masks = self.pad_and_stack(loss_masks_nested, pad_value=0)

        # Truncate to model max length
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        if labels is not None:
            labels = labels[:, :self.tokenizer.model_max_length]
        loss_masks = loss_masks[:, :self.tokenizer.model_max_length]

        # Create standard attention mask (1 for tokens, 0 for padding)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        
        if labels is not None:
            data_batch = dict(
                input_ids=input_ids.long(),
                labels=labels.long(),
                loss_masks=loss_masks,
                attention_mask=attention_mask,
                pad_attention_mask=attention_mask.clone()
            )
            # Create and add custom attention mask if text_part_indices is provided
            if text_part_indices is not None:
                seq_length = input_ids.shape[1]
                custom_attention_mask = self.create_custom_attention_mask(attention_mask, text_part_indices, seq_length)
                data_batch['attention_mask'] = custom_attention_mask
        else:
            data_batch = dict(
                input_ids=input_ids.long(),
                loss_masks=loss_masks,
                attention_mask=attention_mask,
                pad_attention_mask=attention_mask.clone()
            )

        # Handle images if present
        if 'images' in batch[0][0]:
            images = [instance['images'].unsqueeze(0) for sub_batch in batch for instance in sub_batch]
            images = self.pad_and_stack_image_entities(images, pad_value=0)
            data_batch['images'] = images
        
        if "image_masks" in batch[0][0]:
            image_masks = [instance['image_masks'].unsqueeze(0) for sub_batch in batch for instance in sub_batch]
            image_masks = self.pad_and_stack_image_entities(image_masks, pad_value=0)
            data_batch['image_masks'] = image_masks
        
        if "pooled_patches_idx" in batch[0][0]:
            pooled_patches_idx = [instance['pooled_patches_idx'].unsqueeze(0) for sub_batch in batch for instance in sub_batch]
            pooled_patches_idx = self.pad_and_stack_image_entities(pooled_patches_idx, pad_value=-1)
            data_batch['pooled_patches_idx'] = pooled_patches_idx
        
        if "metadata" in batch[0][0]:
            data_batch["metadata"] = [instance["metadata"] for sub_batch in batch for instance in sub_batch]

        return data_batch