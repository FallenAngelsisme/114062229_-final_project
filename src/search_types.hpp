#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include "base_state.hpp"

/* === Parameter map === */
using ParamMap = std::unordered_map<std::string, std::string>;

struct ParamDef {
    enum Type { CHECK, SPIN, COMBO, STRING };
    std::string name;
    Type        type    = CHECK;
    std::string default_val;
    int         min_val = 0;
    int         max_val = 100;
    std::vector<std::string> options;

    ParamDef(std::string n, Type t, std::string def)
        : name(std::move(n)), type(t), default_val(std::move(def)) {}
    ParamDef(std::string n, Type t, std::string def, int mn, int mx)
        : name(std::move(n)), type(t), default_val(std::move(def)), min_val(mn), max_val(mx) {}
};

inline bool param_bool(const ParamMap& m, const std::string& key, bool def){
    auto it = m.find(key);
    if(it == m.end()) return def;
    return (it->second == "true" || it->second == "1");
}
inline int param_int(const ParamMap& m, const std::string& key, int def){
    auto it = m.find(key);
    if(it == m.end()) return def;
    try { return std::stoi(it->second); } catch(...){ return def; }
}

/* === Search result === */
struct RootUpdate {
    Move best_move;
    int  score      = 0;
    int  depth      = 0;
    int  move_index = 0;
    int  total_moves = 0;
};

struct SearchResult {
    Move best_move = {Point(0,0), Point(0,0)};
    int  score     = 0;
    int  depth     = 0;
    int  nodes     = 0;
    int  seldepth  = 0;
    std::vector<Move> pv;
};

/* === Search context (shared state across the search tree) === */
struct SearchContext {
    bool      stop      = false;
    int       nodes     = 0;
    int       seldepth  = 0;
    ParamMap  params;
    std::function<void(RootUpdate)> on_root_update;

    void reset(){
        stop     = false;
        nodes    = 0;
        seldepth = 0;
    }
};

/* Forward declaration needed by minimax.hpp */
class State;
