{ sources ? import ../nix/sources.nix,
  pkgs ? import sources.nixpkgs {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.julia
    pkgs.glibcLocales
  ];
  shellHook = ''
    alias j='${pkgs.julia}/bin/julia --startup-file=no --color=yes'
  '';
}
