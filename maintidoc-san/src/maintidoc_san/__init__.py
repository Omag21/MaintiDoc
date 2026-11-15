"""MaintiDoc SAN — assistant documentaire multimodal pour la maintenance."""

__version__ = "0.2.0"


def main() -> None:
    # Import paresseux : cli.py (et Typer) ne sont chargés que si on lance la commande.
    from .cli import main as _main
    _main()