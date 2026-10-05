{ pkgs ? import <nixpkgs> {} }:
let
  beamline_playground = pkgs.python3Packages.buildPythonPackage {
    pname = "beamline_playground";
    version = "0.1.0";
    pyproject = true;
    src = ./.;
    build-system = [ pkgs.python3Packages.setuptools ];
    dependencies = with pkgs.python3Packages; [ numpy jax pydantic fastapi uvicorn ];
  };
in
pkgs.mkShell {
  inputsFrom = [ beamline_playground ];
  packages = [];
}
