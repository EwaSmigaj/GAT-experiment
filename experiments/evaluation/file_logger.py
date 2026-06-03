from datetime import datetime

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
RESULTS_FILE = f'results/results-{timestamp}.txt'

# Inicjalizuj plik
with open(RESULTS_FILE, 'w') as f:
    f.write('=== WYNIKI TRENINGU ===\n\n')

def log(message):
    with open(RESULTS_FILE, 'a') as f:
        print(message)
        f.write(str(message) + '\n')