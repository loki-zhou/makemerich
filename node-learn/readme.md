安装nodejs的 版本管理工具 nvm
nvm install 18.17.0   # 安装指定版本

nvm use 18.17.0       # 使用指定版本

node -v      # 测试是否正确

npm config set registry https://npmmirror.com/

# 清理缓存并重新尝试安装：
npm cache clean --force
npm install

#  设置代理
npm config set proxy http://proxy.example.com:8080
npm config set https-proxy http://proxy.example.com:8080
