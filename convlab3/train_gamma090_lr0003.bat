@echo off
echo ============================================================
echo  PPO Training - gamma=0.90, lr=0.0003, update_round=10
echo  2000 dialogues/epoch, 20 epochs
echo ============================================================
cd /d "c:\Users\Ahmed\Desktop\HHU REPOs\convlab3\convlab\policy\ppo"
python train.py --config_name=RuleUser-Semantic-RuleDST-gamma090-lr0003 --seed=42
echo.
echo Training complete. Check finished_experiments/ for results.
pause