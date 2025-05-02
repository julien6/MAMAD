#!/usr/bin/env bash

git config --global http.postBuffer 524288000
git config --global http.lowSpeedLimit 0
git config --global http.lowSpeedTime 999999
git config --global core.compression 0

# Check if 'conda' command exists
if [ -d "$HOME/miniconda" ] || [ -d "$HOME/miniconda3" ] ; then
  echo "Miniconda is already installed on your system."
else
  echo "Miniconda is not installed. Installing now..."

  # Download the Miniconda installation script
  curl -o Miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  
  # Make the script executable
  chmod +x Miniconda.sh
  
  # Install Miniconda silently (-b) to the default path ($HOME/miniconda)
  ./Miniconda.sh -b -p "$HOME/miniconda"
  
  # Remove the installer to clean up
  rm -f Miniconda.sh
  
  # Update the PATH to include Miniconda
  export PATH="$HOME/miniconda/bin:$PATH"
  echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc

  source ~/miniconda3/etc/profile.d/conda.sh

  # Verify if conda was successfully installed
  if command -v conda >/dev/null 2>&1; then
    echo "Miniconda was successfully installed."
  else
    echo "Miniconda installation failed. Please check the logs above."
    exit 1
  fi

fi

source ~/miniconda/etc/profile.d/conda.sh

conda create -n mamad python=3.8 -y
conda activate mamad

# Install OvercookedAI
git clone https://github.com/julien6/overcooked_ai.git
cd overcooked_ai
pip install -e .
cd ..

# Install OMARLE environments
git clone https://github.com/julien6/OMARLE.git
cd OMARLE
pip install -e .
cd ..

cd mamad/

pip install -r requirements.txt
