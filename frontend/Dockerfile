# Base image
FROM node:14 as build

# Set the working directory
WORKDIR /frontend

# Copy the package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the frontend code
COPY . .

# Build the frontend
RUN npm run build

EXPOSE 3000

ENV PORT 3000

CMD ["npm", "start"]