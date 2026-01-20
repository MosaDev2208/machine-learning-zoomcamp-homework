# Repository Rename Guide

## From: machine-learning-zoomcamp-homework
## To: machine-learning-zoomcamp

### Step 1: Rename on GitHub (Web Interface)

1. Go to your repository settings:
   ```
   https://github.com/MosaDev2208/machine-learning-zoomcamp-homework/settings
   ```

2. Scroll down to the "Repository name" field

3. Change the name from:
   ```
   machine-learning-zoomcamp-homework
   ```
   to:
   ```
   machine-learning-zoomcamp
   ```

4. Click **"Rename"** button

5. GitHub will automatically redirect to the new URL:
   ```
   https://github.com/MosaDev2208/machine-learning-zoomcamp
   ```

### Step 2: Update Local Repository (Optional but Recommended)

After renaming on GitHub, update your local git remote:

```bash
# Update the remote URL
git remote set-url origin https://github.com/MosaDev2208/machine-learning-zoomcamp.git

# Verify the change
git remote -v
```

### Step 3: Verify the Changes

✅ Check that the new URL works:
```bash
https://github.com/MosaDev2208/machine-learning-zoomcamp
```

✅ Old URL will automatically redirect:
```bash
https://github.com/MosaDev2208/machine-learning-zoomcamp-homework
```

### What's Already Been Updated

✅ **README.md**
- Clone URL updated
- Repository link in footer updated
- All references updated to new name

✅ **Documentation commits**
- Git history preserved
- All commits remain intact

### Notes

- GitHub automatically redirects old URLs for 1 year
- All existing links will continue to work
- No data loss occurs during rename
- All issues, PRs, and wiki links remain the same

### Completion Checklist

- [ ] Rename on GitHub (Settings page)
- [ ] Verify new URL works
- [ ] Update local remote (optional): `git remote set-url origin ...`
- [ ] Push any remaining changes
- [ ] Update any external references if needed

---

**Done!** Your repository is now named `machine-learning-zoomcamp` ✅
