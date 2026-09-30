# ACIT4030 NeRF

Simplified NeRF baseline in PyTorch3D, trained and evaluated on the synthetic **lego** and real **poster** scenes.
Group project for ACIT4030 Machine Learning for 3D Computer Vision, OsloMet.

## Assignment 2

**Due:** 5 October, 23:59

### Tasks

- [ ] Read Chapters 4, 5 and 6 of the course book ([corresponding PyTorch3D tutorials](https://pytorch3d.org/tutorials/))
- [x] Import the Chapter 6 code (a simplified NeRF model) as the baseline for the final project
- [x] Adapt the NeRF tutorial to load the project datasets
- [x] Implement evaluation metrics for quantitative evaluation
- [ ] Write a report (max 10 pages) describing the baseline and presenting qualitative and quantitative results on lego and poster

## Project structure

```
acit4030-nerf/
├── README.md
├── .gitignore
├── requirements.txt
│
├── data/                     # not tracked, extracted acit4030-data.zip
│   ├── lego/
│   └── poster/
│
├── baseline/                 # Chapter 6 code from the course book (cow demo)
│   ├── train_nerf.py
│   ├── nerf_model.py
│   ├── utils/
│   └── data/cow_mesh/        # not tracked, downloaded automatically
│
├── src/                      # our code
│   ├── load_nerf_data.py     # provided loader for lego/poster
│   ├── model.py              # NeRF model
│   ├── train.py              # training on lego/poster
│   ├── evaluate.py           # render test views, compute metrics, save figures
│   └── metrics.py            # PSNR, SSIM, LPIPS
│
├── notebooks/                # exploration and sanity checks
├── outputs/                  # not tracked, checkpoints and renders per dataset
│   ├── lego/
│   └── poster/
│
└── report/
    └── figures/              # figures used in the report
```

## Data

The course dataset (`acit4030-data.zip`) contains:

- **lego**: synthetic scene from the original NeRF paper
- **poster**: real-world scene from nerfstudio
- `load_nerf_data`: utility function that loads both into a PyTorch3D-compatible format

The data is not included in this repository. Extract it into `data/`.

## Training and evaluation

```bash
python src/train.py --dataset lego --n_iter 20000 --batch_size 4
python src/evaluate.py --dataset lego                 # test split
python src/evaluate.py --dataset lego --split train   # optional, to compare with train PSNR
```

Same for `--dataset poster`. The lego test split is `transforms_test.json`; for poster every 8th image is held out.

- `train.py` writes `model.pth`, `config.json`, `losses.json` and intermediate renders to `outputs/<dataset>/`.
- `evaluate.py` writes `metrics_<split>.json` (per-view and mean PSNR/SSIM/LPIPS) and `renders_<split>/` to `outputs/<dataset>/`, and `<dataset>_comparison.png` and `<dataset>_loss.png` to `report/figures/`.
- `batch_size 4` fits an 8 GB GPU; use 2 for 4 GB.

## Running the baseline

```bash
cd baseline
py train_nerf.py
```

The cow mesh is downloaded automatically on the first run.

Notes:
- The original code downloads data with `wget`, which does not exist on Windows. The download was replaced with Python's `urllib`, so the `'wget' is not recognized` messages can be ignored.
- `batch_size` was reduced from 6 to 2 to fit a 4 GB GPU (RTX 3050 Ti).

## Group

- Mats Aakvik Johansen
- Sebastian Skrøvseth Haugen
- Jesse Kyomuhendo Tibamwenda