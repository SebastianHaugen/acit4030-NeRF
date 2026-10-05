# ACIT4030 NeRF

Simplified NeRF baseline in PyTorch3D, trained and evaluated on the synthetic **lego** scene and the real **poster** scene.
Group project for ACIT4030 Machine Learning for 3D Computer Vision at OsloMet.

## Project structure

```
acit4030-nerf/
├── README.md
├── config.yaml                  # all settings for our code
├── requirements.txt             # packages that pip installs
├── .gitignore
├── .gitattributes
│
├── baseline/                    # Chapter 6 code from the course book (cow demo)
│   ├── train_nerf.py
│   ├── nerf_model.py
│   ├── README.md
│   ├── utils/
│   └── data/cow_mesh/           # downloaded automatically on the first run
│
├── data/                        # not tracked, extract acit4030-data.zip here
│   ├── lego/
│   └── poster/
│
├── src/                         # our code
│   ├── repo_util/
│   │   └── LoadConfigurations.py   # reads config.yaml into constants
│   ├── load_nerf_data.py        # provided loader for lego and poster
│   ├── model.py                 # NeRF model from the baseline
│   ├── train.py                 # training on lego or poster
│   ├── smoke_test.py            # short test run before a full training run
│   ├── evaluate.py              # PSNR, SSIM and LPIPS on the test views
│   ├── metrics.py               # the three metrics
│   └── report_figures.py        # loss curves and metrics table for the report
│
├── notebooks/
│   ├── data_check.ipynb                     # checks the datasets and cameras
│   └── training_notebook_for_colab.ipynb    # full pipeline on Google Colab
│
├── outputs/                     # not tracked, results per dataset
│   ├── lego/
│   └── poster/
│
└── report/
    └── figures/                 # figures and tables used in the report
```

## Data

The course dataset `acit4030-data.zip` contains

- **lego**, a synthetic scene from the original NeRF paper
- **poster**, a real-world scene from nerfstudio
- `load_nerf_data`, a function that loads both into a format PyTorch3D can use

The data is not included in this repository. Extract it into `data/` so that you get `data/lego/` and `data/poster/`.

## Installation

PyTorch and PyTorch3D are installed separately, since they depend on the CUDA version. On Colab, the notebook installs PyTorch3D once to Google Drive and reuses it. Everything else is installed with

```bash
pip install -r requirements.txt
```

## Configuration

All settings live in `config.yaml`, and the code reads them through `src/repo_util/LoadConfigurations.py`. The file is never changed during a run. Each run saves a copy of the settings it used as `config_used.yaml` in its output folder.

The dataset is chosen by `active_dataset` in `config.yaml`. To pick a dataset for one run without editing the file, set the `NERF_DATASET` environment variable.

## Running

Run every command from the project root.

```bash
python -m src.smoke_test        # short test run that checks the loss goes down
python -m src.train             # full training on the active dataset
python -m src.evaluate          # metrics and qualitative figure on the test views
python -m src.report_figures    # loss curves and metrics table for both datasets
```

To run the same steps on poster

```bash
NERF_DATASET=poster python -m src.smoke_test
NERF_DATASET=poster python -m src.train
NERF_DATASET=poster python -m src.evaluate
```

In Windows PowerShell, set the variable with `$env:NERF_DATASET="poster"` before the commands instead.

## Running on Google Colab

`notebooks/training_notebook_for_colab.ipynb` runs the whole pipeline for both datasets on Colab. It keeps the dataset, PyTorch3D and all results on your own Google Drive, so nothing is lost when the Colab session ends.

### Google Drive folders

Only the dataset zip has to be uploaded by hand. The notebook creates every other folder itself. As an example, this is how the folders look in Jesse's Drive after a full run.

```
MyDrive/
├── Colab_Packages/                          # created by cell 5, PyTorch3D is installed here once
│   ├── pytorch3d/
│   └── build_info.txt                       # Python, torch and CUDA versions PyTorch3D was built for
│
└── Documents/
    └── ACIT4030_Group_Assignment 2/         # main project folder
        ├── data/
        │   └── acit4030-data.zip            # upload the zip from Canvas here
        ├── outputs/                         # created by cell 9, the results of each run
        │   ├── lego/
        │   └── poster/
        └── report_figures/                  # created by cell 17, a copy of report/figures
```

The `Colab_Packages` folder sits directly in `MyDrive`. Building PyTorch3D takes about 40 minutes, so the notebook only rebuilds it when the Colab versions of Python, torch or CUDA change.

### Steps before the first run

1. Create a project folder on Drive, for example `MyDrive/Documents/ACIT4030_Group_Assignment 2`.
2. Create a `data` folder inside it and upload `acit4030-data.zip` there.
3. Set the paths in cell 1 of the notebook to your own folders. Drive paths in Colab always start with `/content/drive/MyDrive`.

```python
DRIVE_ROOT = "/content/drive/MyDrive/Documents/ACIT4030_Group_Assignment 2"   # your main project folder
DATA_ZIP   = DRIVE_ROOT + "/data/acit4030-data.zip"                         # where you put the zip
PACKAGES   = "/content/drive/MyDrive/Colab_Packages"                        # where PyTorch3D is installed
```

4. Create a GitHub personal access token that can read this repository.
5. In Colab, open the Secrets panel (the key icon on the left), add the token with the name `token_github` and turn on notebook access.
6. Set `BRANCH` in cell 1 to the branch you want to train from.

### Order of the cells

1. Run cells 1 to 4. If cell 4 says PyTorch3D is not usable, set `INSTALL_PYTORCH3D = True` in cell 5, run it once and restart the session.
2. Run cells 6 to 10 to get the code, the packages, the dataset and the links to Drive.
3. Run cells 11 to 13 for lego and cells 14 to 16 for poster. Each dataset starts with a smoke test.
4. Run cell 17 to build the report figures and cell 18 to check that no tracked file was changed.

## Outputs

Each dataset gets its own folder in `outputs/` with

- `checkpoint.pt`, the trained model
- `loss_history.csv`, the losses for every iteration
- `config_used.yaml`, the settings of the run
- `previews/`, renders of one training view during training
- `test_renders/`, every rendered test view
- `metrics_per_view.csv` and `metrics.csv`, the metrics per view and the mean over all views

`report/figures/` gets `qualitative_lego.png`, `qualitative_poster.png`, `loss_curves.pdf` and `table_metrics.tex`.

## Running the original baseline

The code in `baseline/` comes from the Chapter 6 folder of the course book repository, *3D Deep Learning with Python* by Packt.

https://github.com/PacktPublishing/3D-Deep-Learning-with-Python/tree/main/chap6

```bash
cd baseline
py train_nerf.py
```

The cow mesh is downloaded automatically on the first run. Two changes were made to the original code.

- The download used `wget`, which does not exist on Windows, so it was replaced with Python's `urllib`. The `'wget' is not recognized` messages can be ignored.
- `batch_size` was reduced from 6 to 2 to fit a 4 GB GPU (RTX 3050 Ti).

## Group

- Mats Aakvik Johansen
- Sebastian Skrøvseth Haugen
- Jesse Kyomuhendo Tibamwenda