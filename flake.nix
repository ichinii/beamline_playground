{
  description = "2D wave-optics beamline simulation on JAX";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, pyproject-nix }:
    let
      # Single source of truth: dependencies, including the optional extras,
      # are read from pyproject.toml and never duplicated here.
      project = pyproject-nix.lib.project.loadPyproject {
        projectRoot = ./.;
      };

      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = f:
        nixpkgs.lib.genAttrs systems (system:
          f (import nixpkgs { inherit system; }));

      # The library itself. Core dependencies only; the extras ride along as
      # passthru.optional-dependencies so a consumer can opt into them.
      buildLibrary = pythonPackages:
        pythonPackages.buildPythonPackage (
          project.renderers.buildPythonPackage { python = pythonPackages.python; }
        );

      # A python interpreter carrying the library plus any extras requested.
      # Nix has no equivalent of `pip install pkg[server]`, so an extra has to
      # be added to the environment explicitly.
      pythonWith = { pkgs, extras ? [ ] }:
        let
          library = buildLibrary pkgs.python3Packages;
          extraPackages = builtins.concatMap
            (name: library.optional-dependencies.${name} or [ ])
            extras;
        in
        pkgs.python3.withPackages (_: [ library ] ++ extraPackages);
    in
    {
      # Lets other flakes and configurations use the library like any other
      # python package: python3Packages.beamline-playground, with
      # .optional-dependencies.server available when wanted.
      overlays.default = final: prev: {
        pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
          (pythonFinal: _pythonPrev: {
            beamline-playground = buildLibrary pythonFinal;
          })
        ];
      };

      packages = forAllSystems (pkgs:
        let
          # The library on its own. Importing beamline_playground.server from
          # this will fail on purpose: FastAPI is an optional dependency.
          library = buildLibrary pkgs.python3Packages;

          # A python interpreter with the library importable.
          python = pythonWith { inherit pkgs; };

          # The same, plus the server extra, so beamline_playground.server works.
          python-with-server = pythonWith {
            inherit pkgs;
            extras = [ "server" ];
          };

          # Runnable API. Uses `python -m` rather than the package's console
          # script because that script is wrapped with core dependencies only
          # and cannot see the server extra.
          server = pkgs.writeShellScriptBin "beamline-server" ''
            exec ${python-with-server}/bin/python -m beamline_playground.server "$@"
          '';

          # Container image for distributing the server.
          #
          # buildLayeredImage rather than buildImage: the dependency closure is
          # dominated by jax and scipy, which change far less often than this
          # project does, so splitting them into their own layers means a
          # rebuild usually pushes only a small top layer.
          docker = pkgs.dockerTools.buildLayeredImage {
            name = "beamline-playground";
            tag = "latest";
            # Fixed timestamp instead of build time, so identical inputs
            # produce an identical image digest.
            created = "1970-01-01T00:00:01Z";

            contents = [
              server
              # Puts python on PATH so the healthcheck below has an
              # interpreter. Already in the closure, so this costs no space.
              python-with-server
              # /etc/passwd and /etc/group, so running as non-root resolves.
              pkgs.dockerTools.fakeNss
            ];

            extraCommands = ''
              # jax and python want somewhere writable; the image has no /tmp
              # by default and the container runs as a user without a home.
              mkdir -p tmp
              chmod 1777 tmp
            '';

            config = {
              Entrypoint = [ "${server}/bin/beamline-server" ];
              # Bind all interfaces. The CLI defaults to 127.0.0.1, which is
              # correct for local use but unreachable from outside a container.
              Cmd = [ "--host" "0.0.0.0" "--port" "8000" ];
              ExposedPorts = { "8000/tcp" = { }; };
              User = "nobody:nobody";
              Env = [
                "HOME=/tmp"
                "XDG_CACHE_HOME=/tmp"
                "PYTHONDONTWRITEBYTECODE=1"
                "PYTHONUNBUFFERED=1"
              ];
              Labels = {
                "org.opencontainers.image.title" = "beamline-playground";
                "org.opencontainers.image.description" = "2D wave-optics beamline simulation REST API";
                "org.opencontainers.image.version" = "0.1.0";
              };
              # Probes the port baked into Cmd above; override both together.
              Healthcheck = {
                Test = [
                  "CMD"
                  "${python-with-server}/bin/python"
                  "-c"
                  "import urllib.request as r, sys; sys.exit(0 if r.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4).status == 200 else 1)"
                ];
                Interval = 30000000000;
                Timeout = 5000000000;
                StartPeriod = 15000000000;
                Retries = 3;
              };
            };
          };
        in
        {
          inherit library python python-with-server server;
          default = library;
        }
        # dockerTools builds Linux images; it does not work on darwin hosts.
        // pkgs.lib.optionalAttrs pkgs.stdenv.isLinux { inherit docker; });

      apps = forAllSystems (pkgs: rec {
        default = server;
        server = {
          type = "app";
          program = "${self.packages.${pkgs.system}.server}/bin/beamline-server";
          meta.description = "Serve the beamline-playground REST API";
        };
      });

      devShells = forAllSystems (pkgs:
        let
          python = pkgs.python3;
          # Dev shell gets every extra, so the server and the test suite both
          # work without installing the project.
          pythonEnv = python.withPackages (
            project.renderers.withPackages {
              inherit python;
              extras = [ "server" "dev" ];
            }
          );
        in
        {
          default = pkgs.mkShell {
            packages = [ pythonEnv ];
            # Deliberately $PWD and not ${./src}: the latter would copy the
            # sources into the nix store, so edits would not take effect.
            shellHook = ''
              export PYTHONPATH="$PWD/src:$PYTHONPATH"
            '';
          };
        });
    };
}
