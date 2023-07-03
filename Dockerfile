# 设置基础镜像
FROM node:20.2-slim

# 在容器中创建工作目录
WORKDIR /app

# 将 start.sh 文件复制到容器中的 /app 目录下
ADD scaffolds /app/scaffolds
ADD themes /app/themes
COPY start.sh _config.yml package.json /app

RUN npm install 
# 安装依赖
RUN npm install hexo-cli -g
RUN hexo generate
# 暴露Hexo服务器端口，默认为4000
EXPOSE 4000

# 启动Hexo服务器
CMD ["hexo", "server"]
