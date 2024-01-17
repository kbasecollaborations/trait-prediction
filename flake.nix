{
  description = "A Nix-flake-based Python development environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forEachSupportedSystem = f: nixpkgs.lib.genAttrs supportedSystems (system: f {
        pkgs = import nixpkgs { inherit system; };
      });
    in
    {
      devShells = forEachSupportedSystem ({ pkgs }: {
        default = pkgs.mkShell {
          packages = with pkgs; [ python311 virtualenv poetry zlib glibc micromamba ] ++
            (with pkgs.python311Packages; [ pip ]);
	  shellHook = ''
	    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ pkgs.python311 pkgs.virtualenv pkgs.poetry pkgs.zlib pkgs.glibc pkgs.micromamba ]}:$LD_LIBRARY_PATH"
	    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib.outPath}/lib:$LD_LIBRARY_PATH"
	  '';
        };
      });
    };
}
