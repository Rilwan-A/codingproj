# ImputationMethods
Answers to Questions surrounding time series imputation methods

# ImputationMethods
Answers to Questions surrounding time series imputation methods

# Reproducibility
Strongly advise using conda to install tensorflow, instead of pip

# System
- RTX3090
- CUDA 11.8

# Setup
    pip install -r requirements.txt
	export LD_LIBRARY_PATH= {path to cuda libraries}

# Question
### Q1 - Parts a), b), c) 

    bash experiment_scripts/1a/1a_csdis4_v2.sh 0 
    bash experiment_scripts/1a/1a_sssds4_v2.sh 1
    bash experiment_scripts/1a/1a_sssdsa_v2.sh 0

### Q2  - Parts e)

    bash  experiment_scripts/1e.sh 0

### Q2  - Parts f)
    bash  experiment_scripts/1f.sh 0
    
### Q2  - Parts g)
    bash  experiment_scripts/1g.sh 0
