{
  description = "JupyterLab dev shell using Nix Flakes";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = { allowUnfree = true; };
        };

        pythonEnv = pkgs.python311.withPackages (ps: with ps; [
          pip
          pandas
          scikit-learn
          jupyterlab
          matplotlib
        ]);
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
          ];

          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:$LD_LIBRARY_PATH
            if [ ! -d .venv ]; then
              python -m venv .venv
              source .venv/bin/activate
              pip install torch --index-url https://download.pytorch.org/whl/cpu
              pip install torchdata
              pip install matplotlib
              pip install dgl==1.1.3 -f https://data.dgl.ai/wheels/repo.html
              pip install torch-geometric pandas scikit-learn jupyterlab
            else
              source .venv/bin/activate
            fi
          '';
        };
      });
}