# Release Branch Creation Steps

## 🔀 **Release Branch Creation Steps**

### **1. Create and Switch to Release Branch**
```bash
# Create and checkout release branch from main
git checkout -b release/v0.2.0

# Or create branch without switching
git branch release/v0.2.0
```

### **2. Verify Branch Creation**
```bash
# Check current branch
git branch

# Should show:
# * release/v0.2.0
#   main
```

### **3. Final Release Preparation (Optional)**
```bash
# Run final tests to ensure everything works
python -m pytest tests/ -v

# Check code quality one more time
ruff check .
ruff format .
mypy bruno_llm --ignore-missing-imports
```

### **4. Push Release Branch to Remote**
```bash
# Push the new branch to GitHub
git push -u origin release/v0.2.0
```

### **5. Create Release Tag (After Testing)**
```bash
# Create annotated tag for the release
git tag -a v0.2.0 -m "Release v0.2.0: Comprehensive embedding support and bruno-core integration

- Complete Ollama and OpenAI embedding providers
- EmbeddingInterface implementation from bruno-core
- 288+ tests with 89% coverage
- Comprehensive API documentation
- Full backward compatibility with v0.1.0"

# Push tag to remote
git push origin v0.2.0
```

### **6. Create GitHub Release**
1. Go to GitHub repository: `https://github.com/meggy-ai/bruno-llm`
2. Click "Releases" → "Create a new release"
3. Choose tag: `v0.2.0`
4. Release title: `bruno-llm v0.2.0 - Embedding Support & Enhanced Bruno-Core Integration`
5. Copy description from CHANGELOG.md v0.2.0 section
6. Attach distribution files (if built):
   - `dist/bruno_llm-0.2.0-py3-none-any.whl`
   - `dist/bruno_llm-0.2.0.tar.gz`
7. Mark as "Latest release"
8. Click "Publish release"

### **7. Merge Back to Main (After Release)**
```bash
# Switch back to main
git checkout main

# Merge release branch (if any final changes were made)
git merge release/v0.2.0

# Push updated main
git push origin main

# Clean up release branch (optional)
git branch -d release/v0.2.0
git push origin --delete release/v0.2.0
```

## 🚀 **Quick Commands for Your Current State**

Since you're already on main with the v0.2.0 commit ready:

```bash
# 1. Create release branch
git checkout -b release/v0.2.0

# 2. Push to remote
git push -u origin release/v0.2.0

# 3. Create and push tag
git tag -a v0.2.0 -m "Release v0.2.0: Embedding support and bruno-core integration"
git push origin v0.2.0
```

## 📋 **Alternative: Direct Release from Main**

Since your main branch is already clean and ready, you could also:

```bash
# Create tag directly from main
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0

# Then create GitHub release pointing to this tag
```

## 🎯 **When to Use Release Branches**

The release branch approach is recommended for:
- Additional testing in isolation
- Final documentation updates
- Hotfixes if issues are found
- Maintaining a clean release history
- Multiple environments (staging, production)

## 📦 **Building Distribution Packages (Optional)**

If you want to build and test the package locally:

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info/

# Build package
python -m build

# Test installation in clean environment
pip install dist/bruno_llm-0.2.0-py3-none-any.whl

# Verify installation
python -c "from bruno_llm import LLMFactory; from bruno_llm.embedding_factory import EmbeddingFactory; print('✅ Package installed successfully')"
```

## 🚨 **Pre-Release Checklist**

Before creating the release, ensure:

- [ ] All tests pass (`python -m pytest tests/ -v`)
- [ ] Code quality checks pass (`ruff check . && ruff format . && mypy bruno_llm`)
- [ ] Version numbers are updated consistently
- [ ] CHANGELOG.md is updated with release notes
- [ ] Documentation is current and accurate
- [ ] No sensitive information in code or configs
- [ ] All commits are properly signed and clean

## 📋 **Post-Release Tasks**

After successful release:

1. **Update Documentation**
   - Verify documentation links work
   - Update any version references
   - Check that examples still work

2. **Announce Release**
   - GitHub Discussions post
   - Community notifications
   - Update related projects

3. **Monitor Issues**
   - Watch for bug reports
   - Address any immediate issues
   - Plan next release based on feedback

4. **Prepare for Next Development**
   - Create milestone for next version
   - Update project roadmap
   - Begin next feature development
