

# you need to have a config.py that has:
#   DATADIR = # string pointing to the path of the clpsych24 shared task data, e.g., "/data/clp24/" 
#   MODELDIR = # string pointing to a path to a directory where you want to save the finetuning output, e.g., "/data/clp24/finetune-output/"

# 0. Prepare data (see readme, requires some initial steps and processing the official CLPsych data)
python prepare_data.py

# 1. Finetune each model. Our shared task systems only used LMs finetuned on SuicideWatch posts.
python finetune.py --train_group no_risk_sw;
python finetune.py --train_group low_risk_sw;
python finetune.py --train_group moderate_risk_sw;
python finetune.py --train_group high_risk_sw;
python finetune.py --train_group any_risk_sw; 

# 2. (optional) Plot losses.
python plot_loss.py;

# 3. Compute entropies
#   a. compute token entropies from each group model on their own test data
python compute_entropy.py --train_group no_risk_sw --test_group no_risk_sw;
python compute_entropy.py --train_group any_risk_sw --test_group any_risk_sw;
python compute_entropy.py --train_group low_risk_sw --test_group low_risk_sw;
python compute_entropy.py --train_group moderate_risk_sw --test_group moderate_risk_sw;
python compute_entropy.py --train_group high_risk_sw --test_group high_risk_sw;
#   b. compute token entropies from each group model on any risk data
python compute_entropy.py --train_group no_risk_sw --test_group any_risk_sw;
python compute_entropy.py --train_group low_risk_sw --test_group any_risk_sw;
python compute_entropy.py --train_group moderate_risk_sw --test_group any_risk_sw;
python compute_entropy.py --train_group high_risk_sw --test_group any_risk_sw;


# 4. make dataframe of sentence entropies
python sentence_entropy.py

# 5. span selection
python span_selection.py -w # using -w will make the script write submission files with the span selection policies implemented in the script.

# 6. (optional) compare two systems
python compare_systems.py -a max_arch_score_70_submission -b prod_max_arc_ent_ac_70_submission