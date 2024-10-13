# reconstruct4D
This is a 4D reconstruction project.

# setup
clone
```bash
git clone git@github.com:MasahiroOgawa/reconstruct4D.git
git submodule update --init --recursive
```
Install uv for virtual environment management
```
curl -LsSf https://astral.sh/uv/install.sh | sh

```

Set up virtual environment
```
uv sync
source .vev/bin/activate
```

**Notice:** InternImage, and unsupervised_detection will be dropped in the future, so it still use old conda environment.
```
source ${HOME}/anaconda3/bin/activate
conda init
conda env create -f reconstruct4D/ext/InternImage/environment.yml
conda env create -f reconstruct4D/ext/unsupervised_detection/environment.yml
```

# run demo
```
./script/run_foels.sh
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
 cd ../../..
)
git add .
git commit -m "<any comments>"
git push --recurse-submodules=on-demand
(for the first time push, you need to run below;
git push --recurse-submodules=on-demand --set-upstream origin <your branch name>
)
```

## To import submodule upstream update
```
cd ext/<some submodule>
git fetch upstream
git switch main
git merge upstream/main
```