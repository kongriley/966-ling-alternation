# %%
import pymc as pm

with pm.Model() as model:
    p_animate = 0.8
    p_proper = 0.1
    p_complex = 0.9

    animate = pm.Bernoulli('animacy', p=p_animate)
    proper = pm.Bernoulli('proper', p=p_proper)
    complex = pm.Bernoulli('complex', p=p_complex)

    p_of = pm.Deterministic('p_of', 
        pm.math.switch(proper, 0.25,
            pm.math.switch(complex, 
                pm.math.switch(animate, 0, 0.6),
                pm.math.switch(animate, 0.4, 0.9)
            )
        )
    )

    of = pm.Bernoulli('of', p=p_of)

    idata = pm.sample()

# %%

with model:
    idata = pm.sample()
    pm.set_data({'of': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})
    predictions = pm.sample_posterior_predictive(idata, predictions=True).predictions

# %%

predictions.posterior.of.mean(dim=['chain', 'draw'])

# %%
