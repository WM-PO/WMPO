cd ..

bash ./openvla-oft/install_mujoco.sh

# 2. 安装 robosuite
git clone https://github.com/ARISE-Initiative/robosuite.git
cd robosuite
git checkout b9d8d3de5e3dfd1724f4a0e6555246c460407daa
pip install --no-deps -e .
cd ..

# 3. 安装 robomimic
git clone https://github.com/ARISE-Initiative/robomimic.git
cd robomimic
git checkout d0b37cf214bd24fb590d182edb6384333f67b661
pip install --no-deps -e .
cd ..

# 4. 安装 mimicgen
git clone https://github.com/NVlabs/mimicgen.git
cd mimicgen
pip install --no-deps -e .
cd ..

# 5. 安装 robosuite-task-zoo（删除 setup.py 的 install_requires 部分）
git clone https://github.com/ARISE-Initiative/robosuite-task-zoo
cd robosuite-task-zoo
git checkout 74eab7f88214c21ca1ae8617c2b2f8d19718a9ed
# 删除 setup.py 中 install_requires= [...] 部分
# sed -i '/install_requires=/,/],/d' setup.py
pip install --no-deps -e .
cd ..

pip install --no-deps -e .
pip install --no-deps git+https://github.com/moojink/dlimp_openvla
pip install --no-deps "git+https://github.com/moojink/transformers-openvla-oft.git"

echo "installed all dependencies"
