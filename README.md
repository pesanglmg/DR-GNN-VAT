
******************DR-GNN + VAT: Robust Graph-Based Recommendation System**********************

    This project extends the Distributionally Robust Graph Neural Network (DR-GNN) by integrating Virtual Adversarial Training (VAT) to improve robustness against data poisoning and distributional shifts in recommender systems.

******************************* Overview ***************************************
    
    Traditional recommender systems can become unstable when user behaviour changes or when malicious users inject fake data (poisoning attacks).
    This work combines two defence mechanisms:

    Distributionally Robust Optimization (DRO) – protects against worst-case distributional changes (e.g., popularity drift, new user trends).

    Virtual Adversarial Training (VAT) – introduces small, controlled noise in the embedding space during training to make predictions smoother and more stable.

    Together, DRO + VAT help the system stay accurate and reliable, even under adversarial or irregular data conditions.

############################# Repository Structure ################################

    DR-GNN-VAT/
    │
    ├── scripts/
    │   └── poison_train.py          # Generates poisoned dataset (fake users)
    │
    ├── dataloader.py                # Loads and prepares data for training
    ├── model.py                     # DR-GNN + VAT model implementation
    ├── procedure.py                 # Training procedures (BPR loss etc.)
    ├── utils.py, world.py, logger.py# Helper utilities and configs
    ├── requirements.txt             # Dependencies
    └── README.md                    # Project documentation

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Environment Setup ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 1. Clone the repository
    git clone https://github.com/pesanglmg/DR-GNN-VAT.git 
    cd DR-GNN-VAT

    # 2. Create and activate a virtual environment
    python -m venv .venv
    .venv\Scripts\activate   
    

    # 3. Install dependencies
    pip install -r requirements.txt

    \\\\\\\\\\\\\\\\\\\\\\\ Dataset Setup (Gowalla & Gowalla_Poisoned) \\\\\\\\\\\\\\\\\\\\\\

    # 1. Download the processed Gowalla data (from LightGCN baseline)
    git clone --depth=1 https://github.com/gusye1234/LightGCN-PyTorch.git _tmp_lgn
    xcopy /E /I "_tmp_lgn\data\gowalla" "data\gowalla"
    rmdir /S /Q _tmp_lgn

    # 2. Create the poisoned dataset
    python scripts\poison_train.py --path data\gowalla\train.txt ^
    --out data\gowalla_poisoned\train.txt ^
    --target <ITEM_ID> --fraction 0.01 --num_others 9 --seed 2025

    # 3. Build the OOD structure for both datasets
    mkdir OOD_data\popularity_shift\gowalla
    mkdir OOD_data\popularity_shift\gowalla_poisoned
    xcopy /E /I data\gowalla OOD_data\popularity_shift\gowalla
    xcopy /E /I data\gowalla OOD_data\popularity_shift\gowalla_poisoned
    copy /Y data\gowalla_poisoned\train.txt OOD_data\popularity_shift\gowalla_poisoned\train.txt

    $$$$$$$$$$$$$$$$$$$$$$$$$$$$ Training the Model $$$$$$$$$$$$$$$$$$$$$$$$$$$

    Clean data (baseline DR-GNN):

    > python main.py --dataset gowalla --model lgn --epochs 10 ^
    --enable_DRO 1 --aug_on --full_batch --ood popularity_shift ^
    --enable_vat 0

    Poisoned data (DR-GNN + VAT):

    > python main.py --dataset gowalla_poisoned --model lgn --epochs 10 ^
    --enable_DRO 1 --aug_on --full_batch --ood popularity_shift ^
    --enable_vat 1 --vat_coeff 0.1 --vat_eps 1.0 --vat_ip 1

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%% Future Work %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    > Expand evaluation to large datasets (MovieLens 1M, Amazon-Book)

    > Adaptive VAT coefficient scheduling

    > Federated learning integration for privacy preservation

    > Multi-modal extensions (text/image features)

    > Certified robustness analysis

     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  Authors !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        Group 10 — Bachelor of Information Technology
        Charles Darwin University (2025)


    ******************************* Acknowledgement ***********************************
    This project builds upon the original implementation of **DR-GNN (Distributionally Robust Graph Neural Network)** and related graph-based recommendation frameworks shared by the research community.

    The base code, model structure, and dataset configurations were adapted from the open-source repository released by the original DR-GNN authors and the **LightGCN-PyTorch** project.

    We gratefully acknowledge their contributions, as our work extends their efforts by integrating **Virtual Adversarial Training (VAT)** for enhanced robustness against data poisoning and distributional shifts.

    **Original Repositories Referenced:**
    - DR-GNN: [https://github.com/WANGBohaO-jpg/DR-GNN.git]
    - LightGCN: [https://github.com/gusye1234/LightGCN-PyTorch]

    Their pioneering work made this research extension possible.
