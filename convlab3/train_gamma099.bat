@echo off
echo ============================================================
echo  PPO Training - gamma=0.99 (default discount factor)
echo ============================================================
cd /d "c:\Users\Ahmed\Desktop\HHU REPOs\convlab3\convlab\policy\ppo"
python train.py --config_name=RuleUser-Semantic-RuleDST-gamma099 --seed=42
echo.
echo Training complete. Check finished_experiments/ for results.
pause