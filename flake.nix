{
  description = "JupyterLab dev shell using Nix Flakes";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true; # Allows packages like torchWithCuda to be built
          };
        };
        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          ipympl
          jupyterlab
          torch-geometric
          sklearn-compat
          pandas
        ]);
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv
          ];
        };
      });
}

