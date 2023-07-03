# 如果已经有启动的进程则先结束掉
# ps -ef | grep -E "hexo|yarn" | grep -v grep | awk '{print $2}' | xargs kill -9 && \
rm -rf public node_modules *.lock *.json && \
yarn install && \
sleep 10 && \
yarn build && \
sleep 2 && \
hexo server -p 3000