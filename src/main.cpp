// CLI for building, migrating and evaluating HNSW indexes.
#define main benchmark_main
#include "benchmark.cpp"
#undef main

int main(int argc, char **argv) {
    try {
        if (argc < 2 || std::string(argv[1]) == "--help") {
            std::cout << "STARR: Standardized Tolerance for Adaptive Reuse and Repair\n"
                         "Offline HNSW migration (float32 L2)\n"
                         "  build VECTORS DIM N OUTPUT_INDEX\n"
                         "  migrate SOURCE_INDEX SOURCE_DIM TARGET_VECTORS TARGET_DIM N "
                         "OUTPUT_INDEX [TAU B]\n"
                         "  truth VECTORS DIM N OUTPUT_GT\n"
                         "  run METHOD OLD OLD_DIM TARGET TARGET_DIM INDEX GT N\n"
                         "METHOD: Fresh | Ours | DGM-Local | DGM-Search\n";
            return argc < 2 ? 1 : 0;
        }
        const std::string mode = argv[1];
        if (mode != "build" && mode != "migrate")
            return benchmark_main(argc, argv);
        if (mode == "build") {
            check(argc == 6, "build VECTORS DIM N OUTPUT_INDEX");
            const size_t d = std::stoull(argv[3]), n = std::stoull(argv[4]);
            check(d > 0 && n > 0 && n < std::numeric_limits<Id>::max(), "invalid shape");
            check(!std::filesystem::exists(argv[5]), "refusing index overwrite");
            auto v = readRows(argv[2], n, d);
            hnswlib::L2Space space(d);
            // Record allocation-plus-insertion and insertion-only construction times.
            auto start = Clock::now();
            Graph g(&space, n, 16, 200, 100);
            auto insertion_start = Clock::now();
            for (size_t i = 0; i < n; ++i)
                g.addPoint(v[i].data(), i);
            const double insertion_s = elapsed(insertion_start), build_s = elapsed(start);
            inspect(g, &v);
            g.saveIndex(argv[5]);
            std::cout << std::setprecision(17) << "RESULT {\"build_s\":" << build_s
                      << ",\"insertion_s\":" << insertion_s << ",\"n\":" << n << "}\n";
        } else {
            check(argc == 8 || argc == 10,
                  "migrate SOURCE_INDEX SOURCE_DIM TARGET TARGET_DIM N OUTPUT_INDEX [TAU B]");
            const size_t olddim = std::stoull(argv[3]), dim = std::stoull(argv[5]),
                         n = std::stoull(argv[6]);
            check(olddim > 0 && dim > 0 && n > 0 && n < std::numeric_limits<Id>::max(),
                  "invalid shape");
            check(!std::filesystem::exists(argv[7]), "refusing index overwrite");
            auto target = readRows(argv[4], n, dim);
            hnswlib::L2Space oldspace(olddim);
            Graph source(&oldspace, argv[2]);
            check(source.cur_element_count == n, "source size mismatch");
            auto before = inspect(source);
            StableMigrationOptions opts;
            opts.search_ef = 16;
            if (argc == 10) {
                opts.z_tolerance = std::stof(argv[8]);
                opts.search_ef = std::stoull(argv[9]);
            }
            hnswlib::SpaceInterface<float> *ptr = nullptr;
            StableMigrationStats stats;
            auto start = Clock::now();
            auto *migrated = migrateHnswIndexStable(&source, target, &ptr, opts, &stats);
            const double migration_s = elapsed(start);
            Owned output;
            output.space.reset(ptr);
            output.graph.reset(migrated);
            auto audit = inspect(*migrated, &target);
            unchanged(before, inspect(source));
            check(audit.reachable == n, "reachability failure");
            migrated->saveIndex(argv[7]);
            std::cout << std::setprecision(17) << "RESULT {\"migration_s\":" << migration_s
                      << ",\"n\":" << n << ",\"repaired\":" << stats.repaired
                      << ",\"bridges\":" << stats.bridges << ",\"audit\":";
            auditJson(std::cout, audit);
            std::cout << "}\n";
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
