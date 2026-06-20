#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <utility>
#include <cstring>

#include "state.hpp"
#include "minimax.hpp"


/* ============================================================
 * Transposition Table
 * ============================================================ */

enum TTFlag : uint8_t {
    TT_EXACT = 0,
    TT_LOWER = 1,   // alpha (fail-low)  – lower bound
    TT_UPPER = 2,   // beta  (fail-high) – upper bound
};

/* [MODIFIED] 加入 gen 欄位（generation）
 * 原本用 tt_clear() 每步清空 4M entries，非常耗時（50~200ms）
 * 改用 generation 機制：每步只遞增 tt_generation，probe 時檢查 gen 是否一致
 * 舊 generation 的 entry 視為無效，不需要實際清空陣列
 * 節省的時間可以用來搜更深 */
struct TTEntry {
    uint64_t hash   = 0; // 這個盤面的 Zobrist hash（識別用
    int      score  = 0; // 這個盤面算出來的分數
    Move     best_move = {Point(0,0), Point(0,0)}; // 這個盤面最好的步
    int8_t   depth  = 0; // 當時搜尋的深度
    TTFlag   flag   = TT_EXACT; // 分數的類型 exact, alpha(low),beta(high)
    uint32_t gen    = 0;  // [NEW] generation，用來取代清空操作
    bool     valid  = false;  // 這格有沒有存過東西?
};

static const size_t TT_SIZE = 1 << 22;  // ~4M entries (~200 MB) 開一個很大的陣列存所有記錄
static TTEntry tt_table[TT_SIZE];

// [NEW] 每步遞增，讓舊 entry 自動失效，取代清空操作
static uint32_t tt_generation = 0;

static inline size_t tt_index(uint64_t hash){ return hash & (TT_SIZE - 1); } //& (TT_SIZE - 1) 是把 hash 限制在 0 ~ 4百萬之間，概念就像取餘數。

/* [MODIFIED] 原本清空 4M entries（耗時 50~200ms）
 * 改為只遞增 generation，O(1) 時間完成
 * 舊 entry 在 probe 時因 gen 不符而被忽略 */
static void tt_clear(){
    tt_generation++; // [MODIFIED] 只遞增，不清空
    if(tt_generation == 0){
        for(size_t i = 0; i < TT_SIZE; ++i){
            tt_table[i] = TTEntry{};
        }
        tt_generation = 1;
    }
}

//查詢
/* [MODIFIED] 加入 generation 檢查，舊 generation 的 entry 視為無效 */
static const TTEntry* tt_probe(uint64_t hash){
    const TTEntry& e = tt_table[tt_index(hash)];
    if(e.valid && e.hash == hash && e.gen == tt_generation) return &e; // [MODIFIED] 加 gen 檢查
    return nullptr;
}

/* [MODIFIED] 存入時記錄當前 generation */
static void tt_store(uint64_t hash, int score, int depth, TTFlag flag, const Move& best){
    TTEntry& e = tt_table[tt_index(hash)];// 用hash找到位置
    // Always-replace (simplest, good enough)
    e.hash      = hash;
    e.score     = score;
    e.depth     = (int8_t)depth;
    e.flag      = flag;
    e.best_move = best;
    e.gen       = tt_generation; // [MODIFIED] 記錄當前 generation
    e.valid     = true;
}


/* ============================================================
 * [NEW] Time Management
 * search_types.hpp 不能修改，所以時間管理用 static 變數
 * 在 minimax.cpp 內部自行管理，不依賴 SearchContext
 * g_start_time : 這步棋搜尋開始的時間點，在 search() 開頭設定
 * g_time_limit_ms : 這步棋允許的最大搜尋時間（毫秒）
 *   - 從 ctx.params["TimeLimit"] 讀取 CLI 傳入的 --time 參數
 *   - 預設 1800ms，留 200ms 緩衝避免超時被判負
 * ============================================================ */
static std::chrono::steady_clock::time_point g_start_time; // [NEW]
static long g_time_limit_ms = 1800;                         // [NEW]

// [NEW] 回傳已經過的毫秒數，供 Iterative Deepening 判斷要不要開始下一層
static long g_elapsed_ms(){
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - g_start_time).count();
}

// [NEW] 回傳是否超時，供 pvs() 定期呼叫
static bool g_is_time_up(){
    return g_elapsed_ms() >= g_time_limit_ms;
}


static constexpr int MAX_PLY_TRACK = 128;
static Move g_killers[MAX_PLY_TRACK][2];
static int g_history[2][BOARD_H][BOARD_W][BOARD_H][BOARD_W];

static inline bool is_null_move(const Move& mv){
    return mv.first.first == 0 && mv.first.second == 0 && mv.second.first == 0 && mv.second.second == 0;
}

static inline void clear_move_ordering_tables(){
    std::memset(g_killers, 0, sizeof(g_killers));
    std::memset(g_history, 0, sizeof(g_history));
}

static inline void add_killer(int ply, const Move& mv){
    if(ply < 0 || ply >= MAX_PLY_TRACK) return;
    if(mv == g_killers[ply][0]) return;
    g_killers[ply][1] = g_killers[ply][0];
    g_killers[ply][0] = mv;
}

static inline void add_history(int side, const Move& mv, int depth){
    int bonus = depth * depth;
    if(bonus > 256) bonus = 256;
    int& slot = g_history[side][mv.first.first][mv.first.second][mv.second.first][mv.second.second];
    slot += bonus;
    if(slot > 200000) slot = 200000;
}

static inline bool is_quiet_move(const Move& mv, const State* state){
    int tr = (int)mv.second.first;
    int tc = (int)mv.second.second;
    int fr = (int)mv.first.first;
    int fc = (int)mv.first.second;
    int opp = 1 - state->player;
    int captured = state->board.board[opp][tr][tc];
    int mover = state->board.board[state->player][fr][fc];
    bool promo = (mover == 1) && ((state->player == 0 && tr == 0) || (state->player == 1 && tr == BOARD_H - 1));
    return captured == 0 && !promo;
}


/* ============================================================
 * Move Ordering
 * Scores used for std::sort (higher = earlier)
 * ============================================================ */

// MVV-LVA piece values (same scale as config.hpp PIECE_VALUES)
static const int mvv_val[7] = {0, 10, 50, 30, 30, 90, 900};
/*
1. TT move（上一次搜尋找到的最佳步）
2. 吃子（MVV-LVA：用小棋子吃大棋子最優先）
3. 升變
4. 普通移動
*/
static int move_score(const Move& mv, const State* state, const Move& tt_move, int ply){
    // TT move first
    if(mv == tt_move) return 1000000; //上一步

    if(ply >= 0 && ply < MAX_PLY_TRACK){
        if(mv == g_killers[ply][0]) return 900000;
        if(mv == g_killers[ply][1]) return 850000;
    }

    int fr = (int)mv.first.first,  fc = (int)mv.first.second;// 起點
    int tr = (int)mv.second.first, tc = (int)mv.second.second; // 終點
    int opp = 1 - state->player;

    int captured = state->board.board[opp][tr][tc]; // 被吃的棋子
    int mover    = state->board.board[state->player][fr][fc]; // 我方移動的棋子

    if(captured){
        // MVV-LVA: big victim, small attacker
        return 100000 + mvv_val[captured] * 10 - mvv_val[mover];// 吃高價值棋子 用低價值棋子去吃 = 最優先
    }

    // Promotion
    if(mover == 1){
        bool promotes = (state->player == 0 && tr == 0) ||
                        (state->player == 1 && tr == BOARD_H-1);
        if(promotes) return 50000;
    }

    return g_history[state->player][fr][fc][tr][tc];// 普通move
}

static void order_moves(std::vector<Move>& moves, const State* state, const Move& tt_move, int ply){
    std::sort(moves.begin(), moves.end(),
        [&](const Move& a, const Move& b){
            return move_score(a, state, tt_move, ply) > move_score(b, state, tt_move, ply);
        });
}


/* ============================================================
 * Quiescence Search
 * Only expands captures and promotions to avoid horizon effect
 * ============================================================ */
// 因為 quiescence 不限深度，搜到盤面穩定為止。
static int quiescence(
    State* state,
    int alpha,
    int beta,
    int ply,
    SearchContext& ctx,
    const MMParams& p
){
    ctx.nodes++;
    if(ctx.stop) return 0;

    // Stand-pat站立評估
    int stand_pat = state->evaluate(p.use_kp_eval, false, nullptr);//什麼都不做，當前盤面的分數
    if(stand_pat >= beta) return beta; //剪
    if(stand_pat > alpha) alpha = stand_pat; //什麼都不做」比之前找到的步還好

    // Only look at captures / promotions
    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }
    if(state->game_state == WIN) return P_MAX - ply; //目前搜尋到第幾層
    if(state->game_state == DRAW) return 0;

    int opp = 1 - state->player;

    // Collect noisy moves (captures + promotions) noisy moves（吃子＋升變）
    std::vector<Move> noisy;
    noisy.reserve(16);
    for(auto& mv : state->legal_actions){
        int tr = (int)mv.second.first, tc = (int)mv.second.second;
        int captured = state->board.board[opp][tr][tc];
        int mover    = state->board.board[state->player][(int)mv.first.first][(int)mv.first.second];
        bool is_promo = (mover==1) && ((state->player==0 && tr==0)||(state->player==1 && tr==BOARD_H-1));
        if(captured || is_promo) noisy.push_back(mv);
    }

    //Order by MVV-LVA + Promotion
    static const Move null_move = {Point(0,0),Point(0,0)};
    order_moves(noisy, state, null_move, ply);

    for(auto& mv : noisy){
        if(ctx.stop) break;
        State* child = state->next_state(mv);
        if(child->legal_actions.empty() && child->game_state == UNKNOWN){
            child->get_legal_actions();
        }

        int raw;
        if(child->same_player_as_parent()){
            raw = quiescence(child, alpha, beta, ply+1, ctx, p);
        } else {
            raw = -quiescence(child, -beta, -alpha, ply+1, ctx, p);
        }
        delete child;

        if(raw >= beta){ return beta; }
        if(raw > alpha){ alpha = raw; }
    }
    return alpha;
}


/* ============================================================
 * PVS / Alpha-Beta core
 * ============================================================ */

static int pvs(
    State*       state,
    int          depth,
    int          alpha,
    int          beta,
    GameHistory& history,
    int          ply,
    SearchContext& ctx,
    const MMParams& p
){
    ctx.nodes++;
    if(ply > ctx.seldepth) ctx.seldepth = ply; // 記錄最深到幾層
    if(ctx.stop) return 0; // 時間到，立刻停止，ctx.stop在哪裡可以判斷時間到

    /* [NEW] 定期檢查時間
     * 每 4096 個節點（0xFFF + 1）呼叫一次 g_is_time_up()
     * 不是每個節點都查，避免 chrono 呼叫拖慢搜尋速度
     * 原本只在 search() 的迴圈頭檢查時間，
     * 導致某一層搜尋本身很深時，來不及停止而嚴重超時（例如跑了 93 秒）
     * 這裡修正：pvs() 內部也會定期停止 */
    if((ctx.nodes & 0x3FF) == 0){
        if(g_is_time_up()){
            ctx.stop = true;
            return 0;
        }
    }

    /* === Lazy move generation === */ //先查TT → 命中就直接回傳，不用算合法步
    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    /* === Terminal checks === */
    if(state->game_state == WIN){
        return P_MAX - ply;   // prefer faster wins
    }
    if(state->game_state == DRAW){
        return 0;
    }
   
    /* === Repetition === */
    int rep_score;
    if(state->check_repetition(history, rep_score)){
        return rep_score;
    }

    uint64_t h = state->hash();// 不是走過的路徑，棋子的位置

    /* === Transposition table probe === */
    Move tt_best = {Point(0,0), Point(0,0)};
    const TTEntry* tte = tt_probe(h); //查相同局面
    if(tte && tte->depth >= depth){
        int s = tte->score;//depth 夠深 → 直接用分數
        if(tte->flag == TT_EXACT)                   return s; //直接用已知的精確答案，不需要重新計算。
        if(tte->flag == TT_LOWER && s >= beta)      return s; 
        if(tte->flag == TT_UPPER && s <= alpha)     return s;
        tt_best = tte->best_move;
    } else if(tte){ //depth 不夠深，就算深度不同，「這個盤面最好的步」通常還是一樣的
        tt_best = tte->best_move;
    }

    /* === Leaf: quiescence search === */
    if(depth <= 0){
        int score = quiescence(state, alpha, beta, ply, ctx, p);
        return score;
    }

    history.push(h);// 把當前盤面加進歷史記錄

    /* === Move ordering === */
    std::vector<Move> moves = state->legal_actions;
    order_moves(moves, state, tt_best, ply);

    int best_score = M_MAX;
    Move best_move = moves.empty() ? Move{Point(0,0),Point(0,0)} : moves[0];
    int orig_alpha = alpha;
    bool first = true;
    int move_index = 0;

    for(auto& mv : moves){
        if(ctx.stop || g_is_time_up()){
            ctx.stop = true;
            break;
        }

        State* child = state->next_state(mv); 

        int raw;
        bool same = child->same_player_as_parent(); //必不一樣

        if(first){ 
            // Full-window search on first move (PVS)
            raw = pvs(child, depth-1, same ? alpha : -beta, same ? beta : -alpha, history, ply+1, ctx, p);
            if(!same) raw = -raw;
            first = false;//
            //對比第一個 move 用的完整視窗
        } else {
            // Null-window search 極窄視窗 甚麼時候會用到?
            int null_alpha = same ? alpha : -(alpha + 1);//[-(alpha+1), -alpha] 寬度只有 1，非常快。
            int null_beta  = same ? (alpha + 1) : -alpha;

            raw = pvs(child, depth-1, null_alpha, null_beta, history, ply+1, ctx, p);
            if(!same) raw = -raw;

            if(raw > alpha && raw < beta){ // 窄視窗說「這個可能更好」→ 重新用完整視窗搜尋
                // Fail-high on null window: re-search with full window
                raw = pvs(child, depth-1, same ? alpha : -beta, same ? beta : -alpha, history, ply+1, ctx, p);
                if(!same) raw = -raw;
            }
        }
        //raw對手算出的
        delete child;

        if(raw > best_score){
            best_score = raw;
            best_move  = mv;
        }
        if(raw > alpha){
            alpha = raw;
        }
        if(alpha >= beta){
            if(is_quiet_move(mv, state)){
                add_killer(ply, mv);
                add_history(state->player, mv, depth);
            }
            break;  // Beta cut-off
        }
        move_index++;
    }

    history.pop(h);

    if(!ctx.stop){
        TTFlag flag;
        if(best_score <= orig_alpha)   flag = TT_UPPER; // 沒有超過原本的alpha
        else if(best_score >= beta)    flag = TT_LOWER; // 超過beta被剪枝
        else                           flag = TT_EXACT; // 精確值
        tt_store(h, best_score, depth, flag, best_move);
    }

    return best_score;
}


/* ============================================================
 * MiniMax::eval_ctx  (negamax without pruning — kept for compat)
 * ============================================================ */
int MiniMax::eval_ctx(
    State *state,
    int depth,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const MMParams& p
){
    ctx.nodes++;
    if(ply > ctx.seldepth) ctx.seldepth = ply;
    if(ctx.stop) return 0;

    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    // [TODO 3-1] Terminal
    if(state->game_state == WIN)  return P_MAX - ply;
    if(state->game_state == DRAW) return 0;

    int rep_score;
    if(state->check_repetition(history, rep_score)) return rep_score;
    history.push(state->hash());
    //沒到底呢?
    if(depth <= 0){
        int score = state->evaluate(p.use_kp_eval, p.use_eval_mobility, &history);
        history.pop(state->hash());
        return score;
    }

    int best_score = M_MAX;

    for(auto& action : state->legal_actions){
        // [TODO 3-2] create child
        State* next = state->next_state(action);

        bool same = next->same_player_as_parent();

        // [TODO 3-3] search child
        int raw = eval_ctx(next, depth-1, history, ply+1, ctx, p);

        // [TODO 3-4] flip perspective
        int score = same ? raw : -raw;

        delete next;

        // [TODO 3-5] update best
        if(score > best_score) best_score = score;
    }

    history.pop(state->hash());
    return best_score;
}


/* ============================================================
 * MiniMax::search  — Iterative Deepening + PVS
 * ============================================================ */
SearchResult MiniMax::search(
    State *state,
    int depth,
    GameHistory& history,
    SearchContext& ctx
){
    /* [NEW] 從 ctx.params 讀取 CLI 傳入的時間限制
     * CLI --time 2000 會把 2000 存進 ctx.params["TimeLimit"]
     * 留 200ms 緩衝確保不超時被判負
     * 若 params 裡沒有 TimeLimit，預設用 1800ms */
    long raw_time_ms = param_int(ctx.params, "TimeLimit", 2000); // [NEW]
    long capped_time_ms = std::min(raw_time_ms, 1900L);
    g_time_limit_ms  = capped_time_ms - 10;                       // [NEW] 硬上限 1900ms，保留緩衝
    if(g_time_limit_ms < 100) g_time_limit_ms = 100;              // [NEW] 至少 100ms
    g_start_time = std::chrono::steady_clock::now();               // [NEW] 記錄開始時間

    ctx.reset(); // nodes=0, stop=false
    MMParams p = MMParams::from_map(ctx.params); // 讀取設定?
    SearchResult result;
    result.depth = depth;

    if(!state->legal_actions.size()){ //只檢查根節點（第0層）
        state->get_legal_actions();
    }

    if(state->legal_actions.empty()){
        return result;
    }

    /* [NEW] 確保 result 永遠有預設步，防止超時時回傳空步被判負 */
    result.best_move = state->legal_actions[0]; // [NEW]

    tt_clear();
    clear_move_ordering_tables();

    // Iterative deepening
    // [REMOVED] 原本的 start_time、time_limit_ms、time_elapsed_ms lambda
    // 改用 g_start_time、g_time_limit_ms、g_elapsed_ms() 統一管理
    int max_depth = (depth <= 0) ? 999 : depth; // [NEW] depth=0 代表無限深度

    for(int d = 1; d <= max_depth; d++){
        if(ctx.stop) break;

        /* [MODIFIED] 原本：用了 1/2 時間就不開始下一層，導致搜尋深度不夠
         * 改為：用了 3/4 時間才停，讓 AI 有機會搜到更深的層數
         * 分析發現對手能搜到 depth 6-7，而我們常常只搜到 4-5，
         * 搜得深才能做出更好的決策 */
        if(d > 1 && g_elapsed_ms() > g_time_limit_ms - 10){
            // Don't start a deeper iteration if we've used 3/4 of the time
            break;
        }

        ctx.seldepth = 0;
        int best_score = result.score;
        Move best_move = state->legal_actions[0];

        // Order root moves by previous iteration's TT info
        std::vector<Move> root_moves = state->legal_actions;
        {
            Move tt_best = {Point(0,0), Point(0,0)};
            const TTEntry* tte = tt_probe(state->hash());
            if(tte) tt_best = tte->best_move;
            order_moves(root_moves, state, tt_best, 0);
        }

        int total_moves = (int)root_moves.size();
        bool search_aborted = false;

        auto search_root_window = [&](int alpha, int beta, Move& out_move, int& out_score) -> bool {
            int root_alpha = alpha;
            int root_beta  = beta;
            int move_index = 0;
            out_score = M_MAX;
            out_move = root_moves[0];

            for(auto& action : root_moves){
                if(ctx.stop || g_is_time_up()){
                    ctx.stop = true;
                    return false;
                }

                State* child = state->next_state(action);
                bool same = child->same_player_as_parent();

                int raw;
                if(move_index == 0){
                    raw = pvs(child, d-1, same ? root_alpha : -root_beta, same ? root_beta : -root_alpha, history, 1, ctx, p);
                    if(!same) raw = -raw;
                } else {
                    int nw_a = same ? root_alpha : -(root_alpha + 1);
                    int nw_b = same ? (root_alpha + 1) : -root_alpha;
                    raw = pvs(child, d-1, nw_a, nw_b, history, 1, ctx, p);
                    if(!same) raw = -raw;

                    if(raw > root_alpha && raw < root_beta && !ctx.stop){
                        raw = pvs(child, d-1, same ? root_alpha : -root_beta, same ? root_beta : -root_alpha, history, 1, ctx, p);
                        if(!same) raw = -raw;
                    }
                }
                delete child;

                if(raw > out_score){
                    out_score = raw;
                    out_move = action;
                }
                if(raw > root_alpha){
                    root_alpha = raw;
                    if(p.report_partial && ctx.on_root_update){
                        ctx.on_root_update({out_move, out_score, d, move_index+1, total_moves});
                    }
                }
                if(root_alpha >= root_beta){
                    break;
                }
                move_index++;
            }
            return !ctx.stop;
        };

        if(d == 1){
            if(!search_root_window(M_MAX, P_MAX, best_move, best_score)){
                search_aborted = true;
            }
        } else {
            int center = result.score;
            int delta = 40;
            while(true){
                if(ctx.stop || g_is_time_up()){
                    ctx.stop = true;
                    search_aborted = true;
                    break;
                }

                int alpha = std::max(M_MAX, center - delta);
                int beta = std::min(P_MAX, center + delta);
                int score_try = M_MAX;
                Move move_try = root_moves[0];
                if(!search_root_window(alpha, beta, move_try, score_try)){
                    search_aborted = true;
                    break;
                }

                if(score_try <= alpha){
                    center = score_try;
                    delta *= 2;
                    if(delta > 20000) delta = 20000;
                    continue;
                }
                if(score_try >= beta){
                    center = score_try;
                    delta *= 2;
                    if(delta > 20000) delta = 20000;
                    continue;
                }

                best_score = score_try;
                best_move = move_try;
                break;
            }
        }
        
        // Only commit this depth's result if we finished without aborting
        if(!search_aborted && !ctx.stop){
            result.best_move = best_move;
            result.score     = best_score;
            result.depth     = d;
            result.nodes     = ctx.nodes;
            result.pv        = {best_move};
        }

        // If we found a forced win/loss, stop early
        if(std::abs(best_score) >= P_MAX - 100) break; //跳出 Iterative Deepening 的迴圈
    }

    result.nodes = ctx.nodes; 
    return result;// 這次輪到我下棋，我決定走這步」。
}


/* ============================================================
 * default_params / param_defs
 * ============================================================ */
ParamMap MiniMax::default_params(){
    return {
        {"UseKPEval",       "true"},
        {"UseEvalMobility", "true"},
        {"ReportPartial",   "true"},
    };
}

std::vector<ParamDef> MiniMax::param_defs(){
    return {
        {"UseKPEval",       ParamDef::CHECK, "true"},
        {"UseEvalMobility", ParamDef::CHECK, "true"},
        {"ReportPartial",   ParamDef::CHECK, "true"},
    };
}