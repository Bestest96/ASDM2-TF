import tensorflow as tf
import tensorflow.contrib.graph_editor as graph_editor


def duplicate_graph(graph: tf.Tensor, vars_to_replace: dict, name='Duplicated'):
    if graph in vars_to_replace:
        return vars_to_replace[graph]

    ops = []

    def get_ops(t: tf.Tensor):
        if t.op.type != 'VariableV2' and t.op.type != 'Placeholder':
            ops.append(t.op)
            for i in t.op.inputs:
                if i not in vars_to_replace:
                    get_ops(i)

    get_ops(graph)

    sgv = graph_editor.make_view(ops)
    with tf.name_scope(name):
        new_view, _ = graph_editor.copy_with_input_replacements(sgv, vars_to_replace)
        return new_view.outputs[sgv.output_index(graph)]


def _hvp(grad, var, vec):
    return tf.gradients([g * tf.stop_gradient(v) for g, v in zip(grad, vec)], [v for v in var])


def asdm2(loss, lr=1, t0=10.0, delta=0.0005, c=1.0e6, lambda_min=0.5, lambda_max=0.99, eps=1e-8,
          use_nesterov=False, use_scaling=False, scaler_decay=0.999):
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    t_v = tf.Variable(initial_value=tf.constant(1.0),
                      dtype='float',
                      trainable=False)
    beta_v = tf.Variable(initial_value=tf.constant(lr),
                         dtype='float',
                         trainable=False)
    alpha_v = tf.Variable(initial_value=tf.log(beta_v),
                          dtype='float',
                          trainable=False)
    lambda_v = tf.Variable(initial_value=tf.constant(0.99),
                           dtype='float',
                           trainable=False)
    gamma_v = tf.Variable(initial_value=tf.constant(0.99),
                          dtype='float',
                          trainable=False)
    eta_v = tf.Variable(initial_value=-tf.log(tf.constant(1.0) - lambda_v),
                        dtype='float',
                        trainable=False)
    e_dq_db2_v = tf.Variable(initial_value=tf.constant(0.0),
                             dtype='float',
                             trainable=False)
    eg2_v = tf.Variable(initial_value=tf.constant(0.0),
                        dtype='float',
                        trainable=False)
    em2_v = tf.Variable(initial_value=tf.constant(0.0),
                        dtype='float',
                        trainable=False)
    mu_v = tf.Variable(initial_value=tf.constant(0.99),
                       dtype='float',
                       trainable=False)
    nu_v = tf.Variable(initial_value=-tf.log(tf.constant(1.0) - mu_v),
                       dtype='float',
                       trainable=False)
    e_dq_dl2_v = tf.Variable(initial_value=tf.constant(0.0),
                             dtype='float',
                             trainable=False)
    e_dbj_dmu2_v = tf.Variable(initial_value=tf.constant(0.0),
                               dtype='float',
                               trainable=False)
    scaler_v = [tf.Variable(initial_value=tf.ones(v.shape),
                            dtype='float',
                            trainable=False) for v in variables]
    av_g2_v = [tf.Variable(initial_value=tf.ones(v.shape),
                           dtype='float',
                           trainable=False) for v in variables]
    momentum_v = [tf.Variable(initial_value=tf.zeros(v.shape),
                              dtype='float',
                              trainable=False) for v in variables]
    s_dm_db_v = [tf.Variable(initial_value=tf.zeros(v.shape),
                             dtype='float',
                             trainable=False) for v in variables]
    s_dm_dl_v = [tf.Variable(initial_value=tf.zeros(v.shape),
                             dtype='float',
                             trainable=False) for v in variables]
    s_dt_db_v = [tf.Variable(initial_value=tf.zeros(v.shape),
                             dtype='float',
                             trainable=False) for v in variables]
    s_dt_dl_v = [tf.Variable(initial_value=tf.zeros(v.shape),
                             dtype='float',
                             trainable=False) for v in variables]
    phi_v = [tf.Variable(initial_value=tf.zeros(v.shape),
                         dtype='float',
                         trainable=False) for v in variables]
    s_dbt_db_v = [tf.Variable(initial_value=tf.zeros(v.shape),
                              dtype='float',
                              trainable=False) for v in variables]
    s_dbt_dl_v = [tf.Variable(initial_value=tf.zeros(v.shape),
                              dtype='float',
                              trainable=False) for v in variables]
    dbt_dmu_v = [tf.Variable(initial_value=tf.zeros(v.shape),
                             dtype='float',
                             trainable=False) for v in variables]

    grad_var_ns = tf.gradients(xs=variables, ys=loss)
    theta_phi_vars = [v - p if not use_nesterov else v - (p + lambda_v * m)
                      for v, p, m in zip(variables, phi_v, momentum_v)]
    theta_phi_dict = dict(zip([v.op.outputs[0] for v in variables], theta_phi_vars))
    theta_phi_loss = duplicate_graph(loss, theta_phi_dict)
    grad_theta_phi_ns = tf.gradients(xs=theta_phi_vars, ys=theta_phi_loss)

    if use_scaling:
        decay = tf.constant(scaler_decay) * \
                (tf.constant(1.0) - tf.pow(tf.constant(scaler_decay), t_v - tf.constant(1.0))) / \
                (tf.constant(1.0) - tf.pow(tf.constant(scaler_decay), t_v))
        av_g2 = [decay * ag2v + (tf.constant(1.0) - decay) * tf.square(g) for ag2v, g in zip(av_g2_v, grad_var_ns)]
        scaler = [tf.constant(1.0) / tf.sqrt(ag2 + tf.constant(eps)) for ag2 in av_g2]
    else:
        scaler = scaler_v

    s_dg_db_ns = _hvp(grad_var_ns, variables, s_dt_db_v)
    s_dg_dl_ns = _hvp(grad_var_ns, variables, s_dt_dl_v)

    grad_var = [g * s for g, s in zip(grad_var_ns, scaler)]
    grad_theta_phi = [gtp * s for gtp, s in zip(grad_theta_phi_ns, scaler)]
    s_dg_db = [sgb * s for sgb, s in zip(s_dg_db_ns, scaler)]
    s_dg_dl = [sgl * s for sgl, s in zip(s_dg_dl_ns, scaler)]

    dq_db = tf.add_n([tf.reduce_sum(gtp * (sbtb - smb))
                      for gtp, sbtb, smb in zip(grad_theta_phi, s_dbt_db_v, s_dm_db_v)])
    dq_dl = tf.add_n([tf.reduce_sum(gtp * (sbtl - sml))
                      for gtp, sbtl, sml in zip(grad_theta_phi, s_dbt_dl_v, s_dm_dl_v)])
    dbj_dmu = tf.add_n([tf.reduce_sum(gtp * btmu) for gtp, btmu in zip(grad_theta_phi, dbt_dmu_v)])

    cond_t0 = tf.less_equal(t_v, tf.constant(t0))
    grad_norm = tf.sqrt(tf.add_n([tf.reduce_sum(tf.square(g)) for g in grad_var]))
    ag = [hvp * s for hvp, s in zip(_hvp(grad_var, variables, grad_var), scaler)]
    beta = tf.cond(cond_t0,
                   lambda: tf.constant(0.5) / (tf.constant(0.5) * (t_v - tf.constant(1.0)) / (beta_v * t_v) +
                                               tf.sqrt(tf.add_n([tf.reduce_sum(tf.square(hvp))
                                                                 for hvp in ag]))
                                               / (grad_norm * t_v)),
                   lambda: beta_v)
    alpha = tf.cond(cond_t0, lambda: tf.log(beta), lambda: alpha_v)
    lambd = tf.cond(cond_t0, lambda: tf.constant(0.5), lambda: lambda_v)
    eta = tf.cond(cond_t0, lambda: -tf.log(tf.constant(1.0) - lambd), lambda: eta_v)

    cond_neg_t0 = tf.greater(t_v, tf.constant(t0))
    alpha = tf.cond(cond_neg_t0,
                    lambda: alpha - tf.constant(delta) * dq_db / tf.sqrt(e_dq_db2_v),
                    lambda: alpha)
    beta = tf.cond(cond_neg_t0, lambda: tf.exp(alpha), lambda: beta)
    eta = tf.cond(cond_neg_t0,
                  lambda: eta - tf.constant(delta) * dq_dl / tf.sqrt(e_dq_dl2_v),
                  lambda: eta)
    eta = tf.cond(cond_neg_t0,
                  lambda: tf.minimum(tf.maximum(-tf.log(tf.constant(1.0) - tf.constant(lambda_min)), eta),
                                     -tf.log(tf.constant(1.0) - tf.constant(lambda_max))),
                  lambda: eta)
    lambd = tf.cond(cond_neg_t0, lambda: tf.constant(1.0) - tf.exp(-eta), lambda: lambd)
    nu = tf.cond(cond_neg_t0, lambda: nu_v - tf.constant(delta) * dbj_dmu / tf.sqrt(e_dbj_dmu2_v), lambda: nu_v)
    mu = tf.cond(cond_neg_t0, lambda: tf.constant(1.0) - tf.exp(-nu), lambda: mu_v)

    cond_t = tf.greater(t_v, tf.constant(1.0))
    s_dt_db_norm = tf.sqrt(tf.add_n([tf.reduce_sum(tf.square(stb)) for stb in s_dt_db_v]))
    s_dt_dl_norm = tf.sqrt(tf.add_n([tf.reduce_sum(tf.square(stl)) for stl in s_dt_dl_v]))
    alpha0 = tf.cond(cond_t,
                     lambda: tf.log(tf.constant(0.5) *
                                    tf.minimum(s_dt_db_norm / tf.sqrt(tf.add_n([tf.reduce_sum(tf.square(sgb))
                                                                                for sgb in s_dg_db])),
                                               s_dt_dl_norm / tf.sqrt(tf.add_n([tf.reduce_sum(tf.square(sgl))
                                                                                for sgl in s_dg_dl])))),
                     lambda: tf.constant(0.0))
    cond_a0 = tf.greater(alpha, alpha0)
    alpha = tf.cond(cond_t,
                    lambda: tf.cond(cond_a0,
                                    lambda: alpha - tf.constant(2.0) * tf.constant(delta),
                                    lambda: alpha),
                    lambda: alpha)
    beta = tf.cond(cond_t,
                   lambda: tf.cond(cond_a0,
                                   lambda: tf.exp(alpha0),
                                   lambda: beta),
                   lambda: beta)
    gamma = tf.cond(cond_t,
                    lambda: tf.minimum(tf.constant(1.0),
                                       tf.minimum(tf.constant(c) * eg2_v / tf.square(s_dt_db_norm),
                                                  tf.constant(c) * em2_v / tf.square(s_dt_dl_norm))),
                    lambda: gamma_v)
    cond_gl = tf.greater(lambd, gamma)
    eta = tf.cond(cond_t,
                  lambda: tf.cond(cond_gl, lambda: eta - tf.constant(2.0) * tf.constant(delta), lambda: eta),
                  lambda: eta)
    if not use_nesterov:
        s_dbt_db = [mu * gamma * sbtb + (tf.constant(1.0) - mu) * gamma * stb for sbtb, stb in
                    zip(s_dbt_db_v, s_dt_db_v)]
        s_dbt_dl = [mu * gamma * sbtl + (tf.constant(1.0) - mu) * gamma * stl for sbtl, stl in
                    zip(s_dbt_dl_v, s_dt_dl_v)]
        dbt_dmu = [-p + mu * btmu for p, btmu in zip(phi_v, dbt_dmu_v)]
        s_dm_db = [lambd * gamma * smb - g - beta * gamma * sgb for smb, g, sgb in zip(s_dm_db_v, grad_var, s_dg_db)]
        s_dm_dl = [m + lambd * gamma * sml - beta * gamma * sgl for m, sml, sgl in zip(momentum_v, s_dm_dl_v, s_dg_dl)]
        momentum = [lambd * m - beta * g for m, g in zip(momentum_v, grad_var)]
        new_vars = [v + m for v, m in zip(variables, momentum)]
        s_dt_db = [gamma * stb + smb for stb, smb in zip(s_dt_db_v, s_dm_db)]
        s_dt_dl = [gamma * stl + sml for stl, sml in zip(s_dt_dl_v, s_dm_dl)]
    else:
        s_dbt_db = [mu * gamma * sbtb + (tf.constant(1.0) - mu) * gamma * (stb - lambd * smb)
                    for sbtb, stb, smb in zip(s_dbt_db_v, s_dt_db_v, s_dm_db_v)]
        s_dbt_dl = [mu * gamma * sbtl + (tf.constant(1.0) - mu) * (gamma * stl - m - lambd * gamma * sml)
                    for sbtl, stl, m, sml in zip(s_dbt_dl_v, s_dt_dl_v, momentum_v, s_dm_dl_v)]
        dbt_dmu = [-p + mu * btmu for p, btmu in zip(phi_v, dbt_dmu_v)]
        s_dm_db = [lambda_v * gamma * smb - g - beta * gamma * sgb for smb, g, sgb in zip(s_dm_db_v, grad_var, s_dg_db)]
        s_dm_dl = [m + lambda_v * gamma * sml - beta * sgl for m, sml, sgl in
                   zip(momentum_v, s_dm_dl_v, s_dg_dl)]
        momentum = [lambda_v * m - beta * g for m, g in zip(momentum_v, grad_var)]
        new_vars = [v + lambd * m - beta * g for v, m, g in zip(variables, momentum, grad_var)]
        s_dt_db = [gamma * stb + lambd * smb - g - beta * gamma * sgb
                   for stb, smb, g, sgb in zip(s_dt_db_v, s_dm_db, grad_var, s_dg_db)]
        s_dt_dl = [gamma * stl + m + lambd * gamma * sml - beta * gamma * sgl
                   for stl, sml, m, sgl in zip(s_dt_dl_v, s_dm_dl, momentum, s_dg_dl)]
    phi = [mu * p + m for p, m in zip(phi_v, momentum)]

    w_t = tf.constant(0.999) * \
          (tf.constant(1.0) - tf.pow(tf.constant(0.999), t_v - tf.constant(1.0))) / \
          (tf.constant(1.0) - tf.pow(tf.constant(0.999), t_v))
    e_dq_db2 = w_t * e_dq_db2_v + (tf.constant(1.0) - w_t) * tf.square(dq_db)
    e_dq_dl2 = w_t * e_dq_dl2_v + (tf.constant(1.0) - w_t) * tf.square(dq_dl)
    e_dbj_dmu2 = w_t * e_dbj_dmu2_v + (tf.constant(1.0) - w_t) * tf.square(dbj_dmu)
    eg2 = w_t * eg2_v + (tf.constant(1.0) - w_t) * tf.square(grad_norm)
    em2 = w_t * em2_v + (tf.constant(1.0) - w_t) * tf.add_n([tf.reduce_sum(tf.square(m)) for m in momentum_v])

    t = t_v + tf.constant(1.0)

    assignments = [tf.assign(t_v, t), tf.assign(beta_v, beta), tf.assign(lambda_v, lambd), tf.assign(gamma_v, gamma),
                   tf.assign(alpha_v, alpha), tf.assign(eta_v, eta),
                   tf.assign(e_dq_db2_v, e_dq_db2), tf.assign(eg2_v, eg2), tf.assign(em2_v, em2),
                   tf.assign(mu_v, mu), tf.assign(nu_v, nu), tf.assign(e_dq_dl2_v, e_dq_dl2),
                   tf.assign(e_dbj_dmu2_v, e_dbj_dmu2)]
    assignments.extend([tf.assign(v, nv) for v, nv in zip(variables, new_vars)])
    assignments.extend([tf.assign(m_v, m) for m_v, m in zip(momentum_v, momentum)])
    assignments.extend([tf.assign(smb_v, smb) for smb_v, smb in zip(s_dm_db_v, s_dm_db)])
    assignments.extend([tf.assign(sml_v, sml) for sml_v, sml in zip(s_dm_dl_v, s_dm_dl)])
    assignments.extend([tf.assign(stb_v, stb) for stb_v, stb in zip(s_dt_db_v, s_dt_db)])
    assignments.extend([tf.assign(stl_v, stl) for stl_v, stl in zip(s_dt_dl_v, s_dt_dl)])
    assignments.extend([tf.assign(sbtb_v, sbtb) for sbtb_v, sbtb in zip(s_dbt_db_v, s_dbt_db)])
    assignments.extend([tf.assign(sbtl_v, sbtl) for sbtl_v, sbtl in zip(s_dbt_dl_v, s_dbt_dl)])
    assignments.extend([tf.assign(btmu_v, btmu) for btmu_v, btmu in zip(dbt_dmu_v, dbt_dmu)])
    assignments.extend([tf.assign(p_v, p) for p_v, p in zip(phi_v, phi)])
    if use_scaling:
        assignments.extend([tf.assign(ag2v, ag2) for ag2v, ag2 in zip(av_g2_v, av_g2)])
        assignments.extend([tf.assign(sv, s) for sv, s in zip(scaler_v, scaler)])
    return assignments