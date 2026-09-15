# Data access and local layout

This repository does not redistribute dataset videos, captions, checkpoints, or frozen feature caches.

## Official datasets

- **MSR-VTT**: Microsoft Research dataset page: https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/
- **MSVD**: Microsoft download page: https://www.microsoft.com/en-us/download/details.aspx?id=52422
- **VATEX**: official website and annotations: https://eric-xw.github.io/vatex-website/

Some historical web-video URLs may no longer be available. The VATEX protocol should retain only clips that can be decoded, then form disjoint train/development/test video sets as described in the paper.

## Expected local structure

```text
data/
  raw/
    msrvtt/
    msvd/
    vatex/
  processed/
    manifests/
    features/
```

All files below `data/raw/` and `data/processed/` are ignored by Git. Run the manifest builders in `scripts/data/` after downloading the official annotations and videos.
