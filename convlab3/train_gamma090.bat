@echo off
echo ============================================================
echo  PPO Training - gamma=0.90 (lower discount factor)
echo ============================================================
cd /d "c:\Users\Ahmed\Desktop\HHU REPOs\convlab3\convlab\policy\ppo"
python train.py --config_name=RuleUser-Semantic-RuleDST-gamma090 --seed=42
echo.
echo Training complete. Check finished_experiments/ for results.
pause