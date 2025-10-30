#!/bin/bash

# --- Configuration ---
REPO_NAME="parameter_replay_0"
COMMIT_MESSAGE="Auto-sync: update repository files"
# ---

# Check if a .git directory (which defines a git repo) already exists
if [ ! -d ".git" ]; then
    # --- NEW REPO LOGIC ---
    echo "No .git directory found. Initializing new repository..."
    echo "IMPORTANT: This script assumes you have ALREADY created an EMPTY"
    echo "repository on GitHub named '$REPO_NAME'."
    
    # 1. Ask for the user's GitHub username
    read -p "Please enter your GitHub username: " GITHUB_USERNAME
    
    if [ -z "$GITHUB_USERNAME" ]; then
        echo "Username cannot be empty. Exiting."
        exit 1
    fi

    # 2. Construct the remote URL
    REMOTE_URL="https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"

    # 3. Initialize, add, and commit
    git init -b main
    git add .
    git commit -m "Initial commit"
    
    # 4. Link to the remote and push for the first time
    echo "Linking to remote: $REMOTE_URL"
    git remote add origin $REMOTE_URL
    
    echo "Pushing to new repository..."
    git push -u origin main
    
else
    # --- EXISTING REPO LOGIC ---
    echo "Existing .git directory found. Adding, committing, and pushing..."
    
    # 1. Add all new/modified files
    git add .
    
    # 2. Check if there are any changes to commit
    if git diff-index --quiet HEAD --; then
        echo "No changes to commit. Nothing to push."
    else
        # 3. Commit the changes
        echo "Committing changes..."
        git commit -m "$COMMIT_MESSAGE"
        
        # 4. Push the changes to the existing remote
        echo "Pushing to remote..."
        git push
    fi
fi

echo "Script finished-."