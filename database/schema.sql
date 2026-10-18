CREATE TABLE IF NOT EXISTS static_config (
  id                BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (id),  -- forces a single row
  total_drones      SMALLINT NOT NULL CHECK (total_drones > 0),

  -- search area
  latitude          DOUBLE PRECISION NOT NULL,
  longitude         DOUBLE PRECISION NOT NULL,

  -- supplies / restock
  depot_latitude    DOUBLE PRECISION NOT NULL,
  depot_longitude   DOUBLE PRECISION NOT NULL,
  depot_stock       INT NOT NULL CHECK (depot_stock >= 0),          -- total units at depot
  drone_capacity    SMALLINT NOT NULL CHECK (drone_capacity > 0),   -- units per drone
  restock_threshold SMALLINT NOT NULL DEFAULT 0,                    -- return when supplies <= this
  restock_time_s    INT NOT NULL DEFAULT 60,                        -- turnaround at depot

  CHECK (restock_threshold < drone_capacity)
);

CREATE TABLE IF NOT EXISTS input_data (
  drone_id   SMALLINT NOT NULL CHECK (drone_id > 0),
  ts         TIMESTAMPTZ NOT NULL,
  latitude   DOUBLE PRECISION NOT NULL,
  longitude  DOUBLE PRECISION NOT NULL,
  supplies   SMALLINT CHECK (supplies >= 0),                        -- units onboard
  status     TEXT CHECK (status IN ('idle','searching','delivering','restocking','returning')),
  PRIMARY KEY (drone_id, ts)
);

-- Tiger Data: partition input_data by time (replaces the manual ts index)
SELECT create_hypertable('input_data', by_range('ts'), if_not_exists => TRUE);