FROM node:20-alpine

WORKDIR /app

# Copy package manifests and install dependencies
COPY ./frontend/package.json ./frontend/package-lock.json ./
RUN npm install

# Copy application files
COPY ./frontend/ ./

EXPOSE 5173

# Start Vite server
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]

# docker build -f ./frontend/Dockerfile -t pankajmaulekhi/frontend . (from root)
# docker run -p 5173:5173 -e VITE_API_BASE=http://<your-backend-url>:8000 frontend