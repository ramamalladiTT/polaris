git clone https://github.com/huggingface/diffusers.git
echo 'Cloned Diffusers Repo'
cd diffusers
git checkout a58a4f
echo 'Diffusers checkout `a58a4f`'
git apply ../stablediffusion.patch
cp ../run_stablediffusion.py .
echo 'Executing Stable Diffusion simulation'
python3 run_stablediffusion.py
echo 'Simulation runs done!'
