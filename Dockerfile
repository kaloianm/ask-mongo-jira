# Dockerfile for MongoDB 8.0 on Ubuntu 24.04 with Replica Set
#
# Usage:
#   docker build -t ubuntu-mongo:8.0 .
#   docker network create mongo-net
#   docker run -d -p 27017:27017 --name mongo-8.0 --network mongo-net --network-alias mongo-8.0 ubuntu-mongo:8.0
#
# The MongoDB instance will be initialized as a single-node replica set named 'rs0'
# Connect with: mongosh mongodb://mongo-8.0:27017/?replicaSet=rs0

# Use Ubuntu 24.04 as base image
FROM ubuntu:24.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Update package list and install required packages
RUN apt-get update && apt-get upgrade -y
RUN apt-get update && apt-get install -y \
    less \
    vim \
    telnet \
    curl \
    wget \
    gnupg \
    software-properties-common \
    ca-certificates \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Import MongoDB GPG key
RUN curl -fsSL https://www.mongodb.org/static/pgp/server-8.0.asc | \
    gpg -o /usr/share/keyrings/mongodb-server-8.0.gpg --dearmor

# Add MongoDB repository
RUN echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-8.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/8.0 multiverse" | \
    tee /etc/apt/sources.list.d/mongodb-org-8.0.list

# Update package list again and install MongoDB 8.0
RUN apt-get update && apt-get install -y \
    mongodb-org=8.0.* \
    mongodb-org-database=8.0.* \
    mongodb-org-server=8.0.* \
    mongodb-org-mongos=8.0.* \
    mongodb-org-tools=8.0.* \
    mongodb-mongosh \
    && rm -rf /var/lib/apt/lists/*

# Create MongoDB data directory
RUN mkdir -p /data/db && chown -R mongodb:mongodb /data/db

# Create MongoDB log directory
RUN mkdir -p /var/log/mongodb && chown -R mongodb:mongodb /var/log/mongodb

# Expose MongoDB port
EXPOSE 27017

# Set the default user
USER mongodb

# Start MongoDB as a replica set
CMD ["sh", "-c", "mongod --bind_ip 0.0.0.0 --port 27017 --dbpath /data/db --logpath /var/log/mongodb/mongod.log --replSet rs0 & sleep 10 && mongosh --eval 'rs.initiate({_id: \"rs0\", members: [{_id: 0, host: \"localhost:27017\"}]})' && wait"]
