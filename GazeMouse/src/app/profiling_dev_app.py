from cProfile import Profile
from pstats import SortKey, Stats
from src.app.dev_app import run_app

if __name__ == "__main__":
    with Profile() as profile:
        run_app()
        Stats(profile).strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats()
