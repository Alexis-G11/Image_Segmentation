
#!/usr/bin/env bash

# ============================================================================
# IMAGE SEGMENTATION STUDIO
# Complete setup for a 6-member Computer Vision project
# ============================================================================

set -e

PROJECT_NAME="image_segmentation_studio"

echo "=================================================="
echo " Image Segmentation Studio - Project Setup"
echo "=================================================="

# ---------------------------------------------------------------------------
# 1. CREATE / ENTER PROJECT DIRECTORY
# ---------------------------------------------------------------------------

mkdir -p "$PROJECT_NAME"
cd "$PROJECT_NAME"

# Create a file only if it does not exist or is empty.
# Existing non-empty files will not be overwritten.
create_file() {
    local file="$1"

    if [ ! -s "$file" ]; then
        mkdir -p "$(dirname "$file")"
        touch "$file"
        echo "Created: $file"
    else
        echo "Preserved: $file"
    fi
}

# ---------------------------------------------------------------------------
# 2. CREATE TEAM MEMBER WORKSPACES
# ---------------------------------------------------------------------------

echo ""
echo "[1/7] Creating member workspaces..."

mkdir -p members/member1_kmeans/kmeans_results
mkdir -p members/member2_threshold/threshold_results
mkdir -p members/member3_watershed/watershed_results
mkdir -p members/member4_integration
mkdir -p members/member5_testing/test_images
mkdir -p members/member5_testing/charts
mkdir -p members/member6_docs/figures

# ---------------------------------------------------------------------------
# 3. CREATE SHARED PROJECT DIRECTORIES
# ---------------------------------------------------------------------------

echo ""
echo "[2/7] Creating shared directories..."

mkdir -p src
mkdir -p tests/test_images
mkdir -p tests/test_results
mkdir -p docs
mkdir -p results

# ---------------------------------------------------------------------------
# 4. CREATE ALL REQUIRED PYTHON FILES
# ---------------------------------------------------------------------------

echo ""
echo "[3/7] Creating Python files..."

create_file "src/__init__.py"
create_file "src/kmeans_segmentation.py"
create_file "src/thresholding_segmentation.py"
create_file "src/watershed_segmentation.py"
create_file "src/segmentation_studio.py"
create_file "src/utils.py"

create_file "tests/test_scripts.py"
create_file "demo.py"

# ---------------------------------------------------------------------------
# 5. CREATE DOCUMENTATION FILES
# ---------------------------------------------------------------------------

echo ""
echo "[4/7] Creating documentation..."

create_file "docs/report_draft.md"
create_file "docs/contribution_table.md"
create_file "docs/demo_script.md"
create_file "README.md"

# ---------------------------------------------------------------------------
# 6. CREATE GIT CONFIGURATION AND DEPENDENCIES
# ---------------------------------------------------------------------------

echo ""
echo "[5/7] Creating Git configuration and dependencies..."

if [ ! -s ".gitignore" ]; then
cat > .gitignore <<'EOF'
# Python
__pycache__/
*.py[cod]
*.so
*.egg-info/
dist/
build/

# Virtual environments
.venv/
venv/
env/
ENV/

# IDE and operating system
.vscode/
.idea/
*.swp
*.swo
.DS_Store
Thumbs.db

# Generated result images
results/*.png
results/*.jpg
results/*.jpeg
results/*.webp
!results/.gitkeep

# Member-generated outputs
members/*/kmeans_results/*
members/*/threshold_results/*
members/*/watershed_results/*
!members/*/kmeans_results/.gitkeep
!members/*/threshold_results/.gitkeep
!members/*/watershed_results/.gitkeep

# Large media and archives
*.mp4
*.avi
*.zip
*.rar
*.7z

# Jupyter and logs
.ipynb_checkpoints/
*.log
EOF
echo "Created: .gitignore"
else
    echo "Preserved: .gitignore"
fi

if [ ! -s "requirements.txt" ]; then
cat > requirements.txt <<'EOF'
opencv-python>=4.5.0
numpy>=1.19.0
matplotlib>=3.3.0
scikit-learn>=1.0.0
pytest>=7.0.0
EOF
echo "Created: requirements.txt"
else
    echo "Preserved: requirements.txt"
fi

# ---------------------------------------------------------------------------
# 7. CREATE TEAM CONTRIBUTION RULES
# ---------------------------------------------------------------------------

echo ""
echo "[6/7] Creating team contribution guidelines..."

if [ ! -s "CONTRIBUTING.md" ]; then
cat > CONTRIBUTING.md <<'EOF'
# Team Contribution Rules

## 1. Assigned responsibilities

- Member 1: K-means segmentation (M1)
- Member 2: Thresholding segmentation (M2)
- Member 3: Watershed segmentation (M3)
- Member 4: Integration and demo (M4)
- Member 5: Testing and analysis (M5)
- Member 6: Report and documentation (M6)

## 2. Member workspaces

- Member 1: members/member1_kmeans/
- Member 2: members/member2_threshold/
- Member 3: members/member3_watershed/
- Member 4: members/member4_integration/
- Member 5: members/member5_testing/
- Member 6: members/member6_docs/

## 3. Shared code

Members should develop and test their assigned work in their
workspace before submitting it for integration.

The integration lead coordinates changes to src/, demo.py,
requirements.txt, and shared application code.

Testing and documentation changes should be coordinated with
the responsible team members. Do not modify shared files
without communicating with the team.

## 4. Git workflow

1. Check your current branch and working directory.
2. Pull the latest changes before starting work.
3. Work on your assigned branch or files.
4. Test your changes before committing.
5. Commit with a descriptive message.
6. Push your branch and open a pull request if the team
   uses pull requests.
7. Resolve conflicts with the team before merging.

Do not commit passwords, virtual environments, or generated
large output files.

## 5. Project milestones

- M1: K-means segmentation
- M2: Thresholding segmentation
- M3: Watershed segmentation
- M4: Integration and demonstration
- M5: Testing and analysis
- M6: Final report and documentation
EOF
echo "Created: CONTRIBUTING.md"
else
    echo "Preserved: CONTRIBUTING.md"
fi

# ---------------------------------------------------------------------------
# 8. ADD .gitkeep FILES FOR EMPTY DIRECTORIES
# ---------------------------------------------------------------------------

echo ""
echo "[7/7] Adding .gitkeep files..."

for dir in \
    members/member1_kmeans/kmeans_results \
    members/member2_threshold/threshold_results \
    members/member3_watershed/watershed_results \
    members/member5_testing/test_images \
    members/member5_testing/charts \
    members/member6_docs/figures \
    tests/test_images \
    tests/test_results \
    results
do
    create_file "$dir/.gitkeep"
done

# ---------------------------------------------------------------------------
# 9. FINISH
# ---------------------------------------------------------------------------

echo ""
echo "=================================================="
echo " Project setup completed!"
echo "=================================================="
echo ""
echo "Project location:"
pwd
echo ""
echo "Project structure:"
find . -print | sort
echo ""
echo "NEXT STEPS:"
echo "1. Review the project structure."
echo "2. Create a Python virtual environment."
echo "3. Install requirements."
echo "4. Start implementing M1: K-means segmentation."
echo ""
echo "Setup finished successfully!"
