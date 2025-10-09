# Build
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json* pnpm-lock.yaml* yarn.lock* ./
RUN npm ci || npm i
COPY . .
RUN npm run build

# Serve + proxy
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
# Nginx config injects a proxy to backend at http://regulaid:80
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
