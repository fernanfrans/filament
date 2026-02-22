def evaluation_filament(measurement):

    results = {

        # Mean diameter tolerance (less strict)
        'mean_width': 1.65 <= measurement['mean_width'] <= 1.85,

        'std_width': measurement['std_width'] <= 0.30,

        'max_width_change': measurement['max_width_change'] <= 0.40,

        'straight_max_deviation': measurement['straight_max_deviation'] <= 1.5,

        'straight_mean_deviation': measurement['straight_mean_deviation'] <= 0.6
    }

    overall_pass = all(results.values())

    return results, overall_pass