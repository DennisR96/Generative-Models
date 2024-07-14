from diffusers import ScoreSdeVePipeline



model_id = "google/ncsnpp-ffhq-1024"
sde_ve = ScoreSdeVePipeline.from_pretrained(model_id)
