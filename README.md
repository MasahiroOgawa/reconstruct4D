# reconstruct4D
This is a 4D reconstruction project.

# setup
```
conda env create -f script/environment.yml
conda activate reconstruct4D
```

# run demo
```
./script/run_demo.sh
```

# update code
## everyday update for users
```
git pull --recurse-submodules
( or
git pull
git submodule update --init --recursive
)
```

## everyday update for developers
```
(in case you change submodule,
 commit in ext/* directory, and
 cd ../..
 git add ext/
 git submodule update --remote --rebase
)
git add .
git commit -m "<any comments>"
git push --recurse-submodules=on-demand
```