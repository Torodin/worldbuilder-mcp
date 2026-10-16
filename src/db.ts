import Database from "better-sqlite3";

export const db = new Database("world_lore.db");

db.exec(`
    CREATE TABLE IF NOT EXISTS entities (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        type TEXT CHECK( type IN ('CHARACTER', 'FACTION', 'LOCATION', 'RULE')) NOT NULL,
        summary TEXT NOT NULL,
        embedding TEXT NOT NULL,
    );

    CREATE TABLE IF NOT EXISTS relationships (
        source_id TEXT NOT NULL,
        relation_type TEXT CHECK( relation_type IN ('ALLIED_WITH', 'ENEMY_WITH', 'LOCATED_IN', 'BETRAYED')) NOT NULL,
        target_id TEXT NOT NULL,
        notes TEXT,
        PRIMARY KEY (source_id, ralation_type, target_id),
        FOREIGN KEY (source_id) REFERENCES entities(id),
        FOREIGN KEY (target_id) REFERENCES entities(id)
    )
`);
