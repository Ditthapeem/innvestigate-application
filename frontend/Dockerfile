# Base image
FROM node:14 as build

# Set the working directory
WORKDIR /frontend

# Copy the package.json and package-lock.json
COPY package*.json ./

RUN npm install

# Copy the rest of the frontend code
COPY . .

EXPOSE 3000

CMD ["npm", "start"]