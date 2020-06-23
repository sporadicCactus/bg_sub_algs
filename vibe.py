import numpy as np
from itertools import product
from vibe_step import step as _step
from vibe_plus_step import step as _plus_step



class ViBeCommon():
    def __init__(self, step_func, N_samples=20, match_distance=20, min_matches=2, subsample_factor=16, init_interval=10):
        assert(type(N_samples) == int)
        assert(N_samples > 0)
        assert(type(match_distance) == int)
        assert(match_distance > 0)
        assert(type(min_matches) == int)
        assert(min_matches > 0)
        assert(type(subsample_factor) == int)
        assert(subsample_factor > 1)

        #self.input_size = input_size
        self.N = N_samples
        self.R = match_distance
        self.n_min = min_matches
        self.phi = subsample_factor

        self.model = None
        self.init_counter = 0
        self.init_interval = init_interval

        self.step_func = step_func

    def step(self, image):
        image = image.astype(np.uint8)
        
        if self.model is None:
            self.model = image.copy()[:,:,None,:]
        elif self.model.shape[2] < self.N:
            if self.init_counter >= self.init_interval:
                self.model = np.concatenate([self.model, image.copy()[:,:,None,:]], axis=2)
                self.init_counter = 0
            else:
                self.init_counter += 1
        assert(image.shape == (self.model.shape[:2] + (3,)))

        #if self.model is None:
        #    padded_image = np.pad(image, ((1,1),(1,1),(0,0))) 
        #    pixel_neighbourhoods = np.stack([padded_image[1+di:image.shape[0]+1+di, 1+dj:image.shape[1]+1+dj]\
        #        for di,dj in product([-1,0,1],[-1,0,1])], axis=-1)
        #    model = [pixel_neighbourhoods.reshape((image.size, 3*3))[np.arange(image.size), np.random.randint(0, 3*3, image.size)].reshape(image.shape)\
        #    for _ in range(self.N)]
        #    
        #    self.model = np.stack(model, axis=2)

        segmentation_map = np.empty(image.shape[:2], dtype=np.uint8)
        self.step_func(image, self.model, segmentation_map, self.R, self.n_min, self.phi)

        return segmentation_map

class ViBe(ViBeCommon):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, step_func = _step)

class ViBePlus(ViBeCommon):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, step_func=_plus_step)
