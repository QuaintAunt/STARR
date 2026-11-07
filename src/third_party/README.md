# Third-party source

All three dependencies use Apache-2.0. Each directory retains its license;
the DGM directory also retains the upstream NOTICE.

| Directory | Upstream | Source commit | Included files and changes |
| --- | --- | --- | --- |
| `hnswlib/` | [nmslib/hnswlib](https://github.com/nmslib/hnswlib) | [`c1b9b79af3d10c6ee7b5d0afa1ce851ae975254c`](https://github.com/nmslib/hnswlib/tree/c1b9b79af3d10c6ee7b5d0afa1ce851ae975254c) | Seven headers and LICENSE. Source content matches this commit after normalizing CRLF/LF line endings. No algorithm changes. |
| `dgm/` | [misaka0714/DGM-Implement](https://github.com/misaka0714/DGM-Implement) | [`70e31e9d2c681f5872fc9fedffb524f8f4e06caa`](https://github.com/misaka0714/DGM-Implement/tree/70e31e9d2c681f5872fc9fedffb524f8f4e06caa) | Public headers, implementation files, LICENSE and NOTICE. Two files add an optional squared-L2 distance callback; see below. All other included upstream files are unchanged. |
| `dino/` | [facebookresearch/dino](https://github.com/facebookresearch/dino) | [`7c446df5b9f45747937fb0d72314eb9f7b66930a`](https://github.com/facebookresearch/dino/tree/7c446df5b9f45747937fb0d72314eb9f7b66930a) | `hubconf.py`, `utils.py`, `vision_transformer.py` and LICENSE. All four files match this commit byte for byte. |

The DGM changes are in `include/dgm/embedding_space.hpp` and
`src/embedding_space.cpp`. They allow the HNSW adapter to use hnswlib's
squared-L2 kernel. The default DGM distance path is retained. Both files
carry modification notices, and `dgm/distance_callback.patch` records the patch.

The experiment adapters live outside the upstream source directories.
Model weights and corpora are downloaded separately and are not distributed
under this package's source-code license.
