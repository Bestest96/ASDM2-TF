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


def asdm2(loss, eps=1.0, mu=0.5, t0=10.0, delta_alpha=0.0003, delta_eta=0.001, c=1.0e6):
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    t_v = tf.Variable(initial_value=tf.constant(1.0),
                      dtype='float',
                      trainable=False)
    beta_v = tf.Variable(initial_value=tf.constant(eps),
                         dtype='float',
                         trainable=False)
    alpha_v = tf.Variable(initial_value=tf.log(beta_v),
                          dtype='float',
                          trainable=False)
    lambda_v = tf.Variable(initial_value=tf.constant(mu),
                           dtype='float',
                           trainable=False)
    gamma_v = tf.Variable(initial_value=tf.constant(1.0),
                          dtype='float',
                          trainable=False)
    eta_v = tf.Variable(initial_value=-tf.log(tf.constant(1.0) - lambda_v),
                        dtype='float',
                        trainable=False)
    ej_v = tf.Variable(initial_value=tf.constant(0.0),
                       dtype='float',
                       trainable=False)
    e_dj_db2_v = tf.Variable(initial_value=tf.constant(0.0),
                             dtype='float',
                             trainable=False)
    eg2_v = tf.Variable(initial_value=tf.constant(0.0),
                        dtype='float',
                        trainable=False)
    em2_v = tf.Variable(initial_value=tf.constant(0.0),
                        dtype='float',
                        trainable=False)
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
    s_dp_db_v = [tf.Variable(initial_value=tf.zeros(v.shape),
                             dtype='float',
                             trainable=False) for v in variables]
    s_dt_db_avg_v = [tf.Variable(initial_value=tf.zeros(v.shape),
                                 dtype='float',
                                 trainable=False) for v in variables]
    phi_v = [tf.Variable(initial_value=tf.zeros(v.shape),
                         dtype='float',
                         trainable=False) for v in variables]

    grad_var = tf.gradients(xs=variables, ys=loss)
    theta_phi_vars = [v - p for v, p in zip(variables, phi_v)]
    theta_phi_dict = dict(zip([v.op.outputs[0] for v in variables], theta_phi_vars))
    theta_phi_loss = duplicate_graph(loss, theta_phi_dict)
    grad_theta_phi = tf.gradients(xs=theta_phi_vars, ys=theta_phi_loss)

    dj_db = tf.add_n([tf.reduce_sum(g * stb) for g, stb in zip(grad_var, s_dt_db_v)])
    dj_db_avg = tf.add_n([tf.reduce_sum(gtp * stba) for gtp, stba in zip(grad_theta_phi, s_dt_db_avg_v)])

    cond_t0 = tf.less_equal(t_v, tf.constant(t0))
    grad_norm = tf.sqrt(tf.add_n([tf.reduce_sum(tf.square(g)) for g in grad_var]))
    beta = tf.cond(cond_t0,
                   lambda: tf.constant(0.5) / (tf.constant(0.5) * (t_v - tf.constant(1.0)) / (beta_v * t_v) +
                                               tf.sqrt(tf.add_n([tf.reduce_sum(tf.square(hvp))
                                                                 for hvp in
                                                                 _hvp(grad_var, variables, grad_var)]))
                                               / (grad_norm * t_v)),
                   lambda: beta_v)
    alpha = tf.cond(cond_t0, lambda: tf.log(beta), lambda: alpha_v)
    lambd = tf.cond(cond_t0, lambda: tf.constant(0.5), lambda: lambda_v)
    eta = tf.cond(cond_t0, lambda: -tf.log(tf.constant(1.0) - lambd), lambda: eta_v)

    cond_neg_t0 = tf.greater(t_v, tf.constant(t0))
    cond_ej = tf.greater(loss, ej_v)
    alpha = tf.cond(cond_neg_t0,
                    lambda: tf.cond(cond_ej,
                                    lambda: alpha - tf.constant(delta_alpha) * dj_db / tf.sqrt(e_dj_db2_v),
                                    lambda: alpha - tf.constant(delta_alpha) * dj_db_avg / tf.sqrt(e_dj_db2_v)),
                    lambda: alpha)
    beta = tf.cond(cond_neg_t0, lambda: tf.exp(alpha), lambda: beta)

    s_dg_db = _hvp(grad_var, variables, s_dt_db_v)
    s_dg_dl = _hvp(grad_var, variables, s_dt_dl_v)

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
                                    lambda: alpha - tf.constant(10.0) * tf.constant(delta_alpha),
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
                    lambda: tf.constant(0.0))
    cond_gl = tf.greater(lambd, gamma)
    eta = tf.cond(cond_t,
                  lambda: tf.cond(cond_gl, lambda: eta - tf.constant(delta_eta), lambda: eta + tf.constant(delta_eta)),
                  lambda: eta)
    eta = tf.cond(cond_t, lambda: tf.maximum(eta, -tf.log(tf.constant(0.5))), lambda: eta)
    lambd = tf.cond(cond_t, lambda: tf.constant(1.0) - tf.exp(-eta), lambda: lambd)

    ej = lambd * ej_v + (tf.constant(1.0) - lambd) * loss
    w_t = tf.constant(0.999) * \
          (tf.constant(1.0) - tf.pow(tf.constant(0.999), t_v - tf.constant(1.0))) / \
          (tf.constant(1.0) - tf.pow(tf.constant(0.999), t_v))
    e_dj_db2 = w_t * e_dj_db2_v + (tf.constant(1.0) - w_t) * tf.maximum(tf.square(dj_db), tf.square(dj_db_avg))
    eg2 = w_t * eg2_v + (tf.constant(1.0) - w_t) * tf.square(grad_norm)
    em2 = w_t * em2_v + (tf.constant(1.0) - w_t) * tf.add_n([tf.reduce_sum(tf.square(m)) for m in momentum_v])
    s_dm_db = [lambd * gamma * smb - g - beta * gamma * sgb for smb, g, sgb in zip(s_dm_db_v, grad_var, s_dg_db)]
    s_dm_dl = [m + lambd * gamma * sml - beta * gamma * sgl for m, sml, sgl in zip(momentum_v, s_dm_dl_v, s_dg_dl)]
    momentum = [lambd * m - beta * g for m, g in zip(momentum_v, grad_var)]
    new_vars = [v + m for v, m in zip(variables, momentum)]
    s_dt_db = [gamma * stb + smb for stb, smb in zip(s_dt_db_v, s_dm_db)]
    s_dt_dl = [gamma * stl + sml for stl, sml in zip(s_dt_dl_v, s_dm_dl)]
    phi = [gamma * p + m for p, m in zip(phi_v, momentum)]
    s_dp_db = [lambd * gamma * spb + smb for spb, smb in zip(s_dp_db_v, s_dm_db)]
    s_dt_db_avg = [stb - spb for stb, spb in zip(s_dt_db, s_dp_db)]

    t = t_v + tf.constant(1.0)

    assignments = [tf.assign(t_v, t), tf.assign(beta_v, beta), tf.assign(lambda_v, lambd), tf.assign(gamma_v, gamma),
                   tf.assign(alpha_v, alpha), tf.assign(eta_v, eta), tf.assign(ej_v, ej),
                   tf.assign(e_dj_db2_v, e_dj_db2), tf.assign(eg2_v, eg2), tf.assign(em2_v, em2)]
    assignments.extend([tf.assign(v, nv) for v, nv in zip(variables, new_vars)])
    assignments.extend([tf.assign(m_v, m) for m_v, m in zip(momentum_v, momentum)])
    assignments.extend([tf.assign(smb_v, smb) for smb_v, smb in zip(s_dm_db_v, s_dm_db)])
    assignments.extend([tf.assign(sml_v, sml) for sml_v, sml in zip(s_dm_dl_v, s_dm_dl)])
    assignments.extend([tf.assign(stb_v, stb) for stb_v, stb in zip(s_dt_db_v, s_dt_db)])
    assignments.extend([tf.assign(stl_v, stl) for stl_v, stl in zip(s_dt_dl_v, s_dt_dl)])
    assignments.extend([tf.assign(spb_v, spb) for spb_v, spb in zip(s_dp_db_v, s_dp_db)])
    assignments.extend([tf.assign(stba_v, stba) for stba_v, stba in zip(s_dt_db_avg_v, s_dt_db_avg)])
    assignments.extend([tf.assign(p_v, p) for p_v, p in zip(phi_v, phi)])
    return assignments
