## Data setup (Gowalla: clean + poisoned)

We don’t store datasets in the repo. Reproduce them locally:

```bash
# 1) Get processed Gowalla
git clone --depth=1 https://github.com/gusye1234/LightGCN-PyTorch.git _tmp_lgn
xcopy /E /I "_tmp_lgn\data\gowalla" "data\gowalla"
rmdir /S /Q _tmp_lgn

# 2) Create poisoned training file
python scripts\poison_train.py --path data\gowalla\train.txt --out data\gowalla_poisoned\train.txt --target <ITEM_ID> --fraction 0.01 --num_others 9 --seed 2025

# 3) Arrange folders expected by the code
mkdir OOD_data\popularity_shift\gowalla OOD_data\popularity_shift\gowalla_poisoned
xcopy /E /I data\gowalla OOD_data\popularity_shift\gowalla
xcopy /E /I data\gowalla OOD_data\popularity_shift\gowalla_poisoned
copy /Y data\gowalla_poisoned\train.txt OOD_data\popularity_shift\gowalla_poisoned\train.txt
