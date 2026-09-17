"""Comparison arms (baselines) for the meta-diffusion layer.

Each baseline is a ScoreModel subclass swapped in through the model_cls seam of
training.loop.build / train, so the loop, diagnostics, samplers and refinement are reused
unchanged. See baselines/film_conditioning.py for the E15 generic-latent-conditioning arm.
"""
