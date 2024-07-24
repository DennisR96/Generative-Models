def interpolation_lowres(self, tensor, mode):
    tensor_lowres = F.interpolate(
        tensor_hr.unsqueeze(0),
        size=(self.config.dataset.lowres, self.config.dataset.lowres),
        mode='bicubic',
        align_corners=False
    ).squeeze(0)
    
    
    