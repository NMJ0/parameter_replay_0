#!/bin/bash

REPO_NAME="parameter_replay_0"
COMMIT_MESSAGE="Auto-sync: update repository files"

if [ ! -d ".git" ]; then
    echo "No .git directory found. Initializing new repository..."
    read -p "Please enter your GitHub username: " GITHUB_USERNAME
    
    if [ -z "$GITHUB_USERNAME" ]; then
        echo "Username cannot be empty. Exiting."
        exit 1
    fi

    git init -b main
    git add .
    git commit -m "Initial commit"
    
    # Use SSH URL instead of HTTPS
    REMOTE_URL="git@github.com:$GITHUB_USERNAME/$REPO_NAME.git"
    echo "Linking to remote: $REMOTE_URL"
    git remote add origin $REMOTE_URL
    
    echo "Pushing to new repository..."
    git push -u origin main
    
else
    echo "Existing .git directory found. Adding, committing, and pushing..."
    
    git add .
    
    if git diff-index --quiet HEAD --; then
        echo "No changes to commit. Nothing to push."
    else
        echo "Committing changes..."
        git commit -m "$COMMIT_MESSAGE"
        
        echo "Pushing to remote..."
        git push 2>&1
        
        if [ $? -ne 0 ]; then
            echo ""
            echo "❌ Push failed. Possible solutions:"
            echo "1. SSH: Set up SSH keys and use 'git@github.com:...' URLs"
            echo "2. Token: Use GitHub Personal Access Token instead of password"
            echo "3. Check: Verify credentials at https://github.com/settings/tokens"
            exit 1
        fi
    fi
fi

echo "✓ Script finished."
