# pip install git+https://github.com/fangqi-Zhu/ColossalAI.git
# pip install git+https://github.com/fangqi-Zhu/TensorNVMe.git
# pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/fangqi-Zhu/apex.git
pip install webdataset==0.2.111 scikit-learn
pip install -v -e .

pip install imageio

cd ../openvla
pip install -e .
